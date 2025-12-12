#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "apriltag.h"
#include "apriltag_pose.h"
#include "tag16h5.h"
#include "tag25h9.h"
#include "tag36h10.h"
#include "tag36h11.h"
#include "tagCircle21h7.h"
#include "tagCircle49h12.h"
#include "tagCustom48h12.h"
#include "tagStandard41h12.h"
#include "tagStandard52h13.h"

#include <cstring>
#include <mutex>
#include <stdexcept>

namespace nb = nanobind;

using GrayImage =
    nb::ndarray<uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

struct Detector {
  apriltag_detector_t *td = nullptr;
  apriltag_family_t *tf = nullptr;
  void (*destroy_func)(apriltag_family_t *) = nullptr;
  std::mutex det_lock;

  Detector(const char *family, int threads, int maxhamming, float decimate,
           float blur, bool refine_edges, bool debug) {
    // Create family
    if (strcmp(family, "tag36h11") == 0) {
      tf = tag36h11_create();
      destroy_func = tag36h11_destroy;
    } else if (strcmp(family, "tag36h10") == 0) {
      tf = tag36h10_create();
      destroy_func = tag36h10_destroy;
    } else if (strcmp(family, "tag25h9") == 0) {
      tf = tag25h9_create();
      destroy_func = tag25h9_destroy;
    } else if (strcmp(family, "tag16h5") == 0) {
      tf = tag16h5_create();
      destroy_func = tag16h5_destroy;
    } else if (strcmp(family, "tagCircle21h7") == 0) {
      tf = tagCircle21h7_create();
      destroy_func = tagCircle21h7_destroy;
    } else if (strcmp(family, "tagCircle49h12") == 0) {
      tf = tagCircle49h12_create();
      destroy_func = tagCircle49h12_destroy;
    } else if (strcmp(family, "tagStandard41h12") == 0) {
      tf = tagStandard41h12_create();
      destroy_func = tagStandard41h12_destroy;
    } else if (strcmp(family, "tagStandard52h13") == 0) {
      tf = tagStandard52h13_create();
      destroy_func = tagStandard52h13_destroy;
    } else if (strcmp(family, "tagCustom48h12") == 0) {
      tf = tagCustom48h12_create();
      destroy_func = tagCustom48h12_destroy;
    } else {
      throw std::runtime_error(
          std::string("Unrecognized tag family: '") + family +
          "'. Supported families: tag36h11, tag36h10, tag25h9, tag16h5, "
          "tagCircle21h7, tagCircle49h12, tagStandard41h12, "
          "tagStandard52h13, tagCustom48h12");
    }

    td = apriltag_detector_create();
    if (!td) {
      destroy_func(tf);
      throw std::runtime_error("Failed to create detector");
    }

    errno = 0;
    apriltag_detector_add_family_bits(td, tf, maxhamming);

    if (errno == EINVAL) {
      apriltag_detector_destroy(td);
      destroy_func(tf);
      throw std::runtime_error("maxhamming parameter should not exceed 3");
    }
    if (errno == ENOMEM) {
      apriltag_detector_destroy(td);
      destroy_func(tf);
      throw std::runtime_error("Insufficient memory for tag family decoder");
    }

    td->quad_decimate = decimate;
    td->quad_sigma = blur;
    td->nthreads = threads;
    td->refine_edges = refine_edges;
    td->debug = debug;
  }

  ~Detector() {
    if (td)
      apriltag_detector_destroy(td);
    if (tf && destroy_func)
      destroy_func(tf);
  }

  nb::list detect(GrayImage image) {
    int height = static_cast<int>(image.shape(0));
    int width = static_cast<int>(image.shape(1));
    int stride = static_cast<int>(image.stride(0));

    image_u8_t im = {.width = width,
                     .height = height,
                     .stride = stride,
                     .buf = const_cast<uint8_t *>(image.data())};

    zarray_t *detections;
    {
      nb::gil_scoped_release release;
      std::lock_guard<std::mutex> lock(det_lock);
      detections = apriltag_detector_detect(td, &im);
    }

    if (!detections) {
      if (errno == EAGAIN) {
        throw std::runtime_error("Unable to create detector threads");
      }
      return nb::list();
    }

    nb::list result;
    int n = zarray_size(detections);

    for (int i = 0; i < n; i++) {
      apriltag_detection_t *det;
      zarray_get(detections, i, &det);

      // Center array
      auto center = nb::ndarray<nb::numpy, double, nb::shape<2>>(
          new double[2]{det->c[0], det->c[1]}, {2},
          nb::capsule(new double[2], [](void *p) noexcept {
            delete[] static_cast<double *>(p);
          }));

      // Corners array
      double *corners_data = new double[8];
      for (int j = 0; j < 4; j++) {
        corners_data[j * 2] = det->p[j][0];
        corners_data[j * 2 + 1] = det->p[j][1];
      }
      auto corners = nb::ndarray<nb::numpy, double, nb::shape<4, 2>>(
          corners_data, {4, 2}, nb::capsule(corners_data, [](void *p) noexcept {
            delete[] static_cast<double *>(p);
          }));

      // Homography matrix
      double *H_data = new double[9];
      for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
          H_data[r * 3 + c] = MATD_EL(det->H, r, c);
        }
      }
      auto H = nb::ndarray<nb::numpy, double, nb::shape<3, 3>>(
          H_data, {3, 3}, nb::capsule(H_data, [](void *p) noexcept {
            delete[] static_cast<double *>(p);
          }));

      nb::dict d;
      d["id"] = det->id;
      d["hamming"] = det->hamming;
      d["margin"] = det->decision_margin;
      d["center"] = center;
      d["lb-rb-rt-lt"] = corners;
      d["H"] = H;

      result.append(d);
    }

    apriltag_detections_destroy(detections);
    return result;
  }

  nb::dict estimate_tag_pose(nb::dict detection, nb::dict info) {
    // Parse info
    double tag_size = nb::cast<double>(info["tagsize"]);
    double fx = nb::cast<double>(info["fx"]);
    double fy = nb::cast<double>(info["fy"]);
    double cx = nb::cast<double>(info["cx"]);
    double cy = nb::cast<double>(info["cy"]);

    // Get homography
    auto H_arr = nb::cast<nb::ndarray<double, nb::shape<3, 3>, nb::c_contig>>(
        detection["H"]);
    const double *H_data = H_arr.data();

    // Build detection struct
    apriltag_detection_t det;
    memset(&det, 0, sizeof(det));

    // Get corners
    auto corners = nb::cast<nb::ndarray<double, nb::shape<4, 2>, nb::c_contig>>(
        detection["lb-rb-rt-lt"]);
    const double *corner_data = corners.data();
    for (int i = 0; i < 4; i++) {
      det.p[i][0] = corner_data[i * 2];
      det.p[i][1] = corner_data[i * 2 + 1];
    }

    // Copy homography
    matd_t *H = matd_create(3, 3);
    for (int r = 0; r < 3; r++) {
      for (int c = 0; c < 3; c++) {
        MATD_EL(H, r, c) = H_data[r * 3 + c];
      }
    }
    det.H = H;

    // Get center if available
    if (detection.contains("center")) {
      auto center_arr =
          nb::cast<nb::ndarray<double, nb::shape<2>, nb::c_contig>>(
              detection["center"]);
      const double *center_data = center_arr.data();
      det.c[0] = center_data[0];
      det.c[1] = center_data[1];
    }

    // Estimate pose
    apriltag_detection_info_t det_info = {.det = &det,
                                          .tagsize = tag_size,
                                          .fx = fx,
                                          .fy = fy,
                                          .cx = cx,
                                          .cy = cy};

    apriltag_pose_t pose;
    double err = ::estimate_tag_pose(&det_info, &pose);

    matd_destroy(H);

    // Create output arrays
    double *R_data = new double[9];
    for (int r = 0; r < 3; r++) {
      for (int c = 0; c < 3; c++) {
        R_data[r * 3 + c] = MATD_EL(pose.R, r, c);
      }
    }
    auto R = nb::ndarray<nb::numpy, double, nb::shape<3, 3>>(
        R_data, {3, 3}, nb::capsule(R_data, [](void *p) noexcept {
          delete[] static_cast<double *>(p);
        }));

    double *t_data = new double[3];
    for (int i = 0; i < 3; i++) {
      t_data[i] = MATD_EL(pose.t, i, 0);
    }
    auto t = nb::ndarray<nb::numpy, double, nb::shape<3>>(
        t_data, {3}, nb::capsule(t_data, [](void *p) noexcept {
          delete[] static_cast<double *>(p);
        }));

    matd_destroy(pose.R);
    matd_destroy(pose.t);

    nb::dict result;
    result["R"] = R;
    result["t"] = t;
    result["e"] = err;
    return result;
  }
};

NB_MODULE(apriltag, m) {
  m.doc() = "AprilTag visual fiducial system detector";

  nb::class_<Detector>(
      m, "apriltag",
      "AprilTag detector.\n\n"
      "Parameters\n"
      "----------\n"
      "family : str\n"
      "    Tag family name (e.g., 'tag36h11', 'tag25h9').\n"
      "threads : int, default=1\n"
      "    Number of threads to use.\n"
      "maxhamming : int, default=1\n"
      "    Maximum hamming distance for error correction (0-3).\n"
      "decimate : float, default=2.0\n"
      "    Decimation factor for speed.\n"
      "blur : float, default=0.0\n"
      "    Gaussian blur sigma.\n"
      "refine_edges : bool, default=True\n"
      "    Refine tag edges.\n"
      "debug : bool, default=False\n"
      "    Enable debug output.\n")
      .def(nb::init<const char *, int, int, float, float, bool, bool>(),
           nb::arg("family"), nb::arg("threads") = 1, nb::arg("maxhamming") = 1,
           nb::arg("decimate") = 2.0f, nb::arg("blur") = 0.0f,
           nb::arg("refine_edges") = true, nb::arg("debug") = false)
      .def("detect", &Detector::detect, nb::arg("image"),
           "Detect AprilTags in a grayscale image.\n\n"
           "Parameters\n"
           "----------\n"
           "image : numpy.ndarray\n"
           "    Grayscale image (uint8, 2D array).\n\n"
           "Returns\n"
           "-------\n"
           "list of dict\n"
           "    Each dict contains: 'id', 'hamming', 'margin', 'center',\n"
           "    'lb-rb-rt-lt' (corners), 'H' (homography).\n")
      .def("estimate_tag_pose", &Detector::estimate_tag_pose,
           nb::arg("detection"), nb::arg("info"),
           "Estimate 3D pose of a detected tag.\n\n"
           "Parameters\n"
           "----------\n"
           "detection : dict\n"
           "    Detection from detect().\n"
           "info : dict\n"
           "    Camera info with keys: 'tagsize', 'fx', 'fy', 'cx', 'cy'.\n\n"
           "Returns\n"
           "-------\n"
           "dict\n"
           "    Pose with keys: 'R' (3x3 rotation), 't' (3D translation),\n"
           "    'e' (error).\n");
}
