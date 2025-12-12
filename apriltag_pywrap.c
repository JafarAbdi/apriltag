#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <stdbool.h>
#include <Python.h>
#ifndef Py_PYTHREAD_H
#include <pythread.h>
#endif
#include <structmember.h>
#include <numpy/arrayobject.h>
#include <signal.h>

#include "apriltag.h"
#include "apriltag_pose.h"
#include "tag36h10.h"
#include "tag36h11.h"
#include "tag25h9.h"
#include "tag16h5.h"
#include "tagCircle21h7.h"
#include "tagCircle49h12.h"
#include "tagCustom48h12.h"
#include "tagStandard41h12.h"
#include "tagStandard52h13.h"


#define SUPPORTED_TAG_FAMILIES(_)           \
    _(tag36h10)                             \
    _(tag36h11)                             \
    _(tag25h9)                              \
    _(tag16h5)                              \
    _(tagCircle21h7)                        \
    _(tagCircle49h12)                       \
    _(tagStandard41h12)                     \
    _(tagStandard52h13)                     \
    _(tagCustom48h12)

#define TAG_CREATE_FAMILY(name) \
    else if (0 == strcmp(family, #name)) self->tf = name ## _create();
#define TAG_SET_DESTROY_FUNC(name) \
    else if (0 == strcmp(family, #name)) self->destroy_func = name ## _destroy;
#define FAMILY_STRING(name) "  " #name "\n"


// Python is silly. There's some nuance about signal handling where it sets a
// SIGINT (ctrl-c) handler to just set a flag, and the python layer then reads
// this flag and does the thing. Here I'm running C code, so SIGINT would set a
// flag, but not quit, so I can't interrupt the solver. Thus I reset the SIGINT
// handler to the default, and put it back to the python-specific version when
// I'm done
#define SET_SIGINT() struct sigaction sigaction_old;                    \
do {                                                                    \
    if( 0 != sigaction(SIGINT,                                          \
                       &(struct sigaction){ .sa_handler = SIG_DFL },    \
                       &sigaction_old) )                                \
    {                                                                   \
        PyErr_SetString(PyExc_RuntimeError, "sigaction() failed");      \
        goto done;                                                      \
    }                                                                   \
} while(0)
#define RESET_SIGINT() do {                                             \
    if( 0 != sigaction(SIGINT,                                          \
                       &sigaction_old, NULL ))                          \
        PyErr_SetString(PyExc_RuntimeError, "sigaction-restore failed"); \
} while(0)

#define PYMETHODDEF_ENTRY(function_prefix, name, args) {#name,          \
                                                        (PyCFunction)function_prefix ## name, \
                                                        args,           \
                                                        function_prefix ## name ## _docstring}

typedef struct {
    PyObject_HEAD

    apriltag_family_t*   tf;
    apriltag_detector_t* td;
    PyThread_type_lock   det_lock;
    void (*destroy_func)(apriltag_family_t *tf);
} apriltag_py_t;


static PyObject *
apriltag_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    errno = 0;

    bool success = false;

    apriltag_py_t* self = (apriltag_py_t*)type->tp_alloc(type, 0);
    if(self == NULL) goto done;

    self->tf = NULL;
    self->td = NULL;

    self->det_lock = PyThread_allocate_lock();
    if (self->det_lock == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to allocate detection lock");
        goto done;
    }

    const char* family          = NULL;
    int         Nthreads        = 1;
    int         maxhamming      = 1;
    float       decimate        = 2.0;
    float       blur            = 0.0;
    bool        refine_edges    = true;
    bool        debug           = false;
    PyObject*   py_refine_edges = NULL;
    PyObject*   py_debug        = NULL;

    char* keywords[] = {"family",
                        "threads",
                        "maxhamming",
                        "decimate",
                        "blur",
                        "refine_edges",
                        "debug",
                        NULL };

    if(!PyArg_ParseTupleAndKeywords( args, kwargs, "s|iiffOO",
                                     keywords,
                                     &family,
                                     &Nthreads,
                                     &maxhamming,
                                     &decimate,
                                     &blur,
                                     &py_refine_edges,
                                     &py_debug ))
    {
        goto done;
    }

    if(py_refine_edges != NULL)
        refine_edges = PyObject_IsTrue(py_refine_edges);
    if(py_debug        != NULL)
        debug        = PyObject_IsTrue(py_debug);


    if(0) ; SUPPORTED_TAG_FAMILIES(TAG_SET_DESTROY_FUNC)
    else
    {
        PyErr_Format(PyExc_RuntimeError, "Unrecognized tag family name: '%s'. Families I know about:\n%s",
                     family, SUPPORTED_TAG_FAMILIES(FAMILY_STRING));
        goto done;
    }

    if(0) ; SUPPORTED_TAG_FAMILIES(TAG_CREATE_FAMILY);

    self->td = apriltag_detector_create();
    if(self->td == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "apriltag_detector_create() failed!");
        goto done;
    }

    apriltag_detector_add_family_bits(self->td, self->tf, maxhamming);
    self->td->quad_decimate       = decimate;
    self->td->quad_sigma          = blur;
    self->td->nthreads            = Nthreads;
    self->td->refine_edges        = refine_edges;
    self->td->debug               = debug;

    switch(errno){
        case EINVAL:
                PyErr_SetString(PyExc_RuntimeError, "Unable to add family to detector. \"maxhamming\" parameter should not exceed 3");
                break;
        case ENOMEM:
                PyErr_Format(PyExc_RuntimeError, "Unable to add family to detector due to insufficient memory to allocate the tag-family decoder. Try reducing \"maxhamming\" from %d or choose an alternative tag family",maxhamming);
                break;
        default:
            success = true;
    }

 done:
    if(!success)
    {
        if(self != NULL)
        {
            if(self->td != NULL)
            {
                apriltag_detector_destroy(self->td);
                self->td = NULL;
            }
            if(self->tf != NULL)
            {
                self->destroy_func(self->tf);
                self->tf = NULL;
            }
            if(self->det_lock != NULL)
            {
                PyThread_free_lock(self->det_lock);
                self->det_lock = NULL;
            }
            Py_DECREF(self);
        }
        return NULL;
    }

    return (PyObject*)self;
}

static void apriltag_dealloc(apriltag_py_t* self)
{
    if(self == NULL)
        return;
    if(self->td != NULL)
    {
        apriltag_detector_destroy(self->td);
        self->td = NULL;
    }
    if(self->tf != NULL)
    {
        self->destroy_func(self->tf);
        self->tf = NULL;
    }
    if(self->det_lock != NULL)
    {
        PyThread_free_lock(self->det_lock);
        self->det_lock = NULL;
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* apriltag_detect(apriltag_py_t* self,
                                 PyObject* args)
{
    errno = 0;

    PyObject*      result           = NULL;
    PyArrayObject* xy_c             = NULL;
    PyArrayObject* xy_lb_rb_rt_lt   = NULL;
    PyArrayObject* image            = NULL;
    PyObject*      detections_tuple = NULL;

#ifdef _POSIX_C_SOURCE
    SET_SIGINT();
#endif
    if(!PyArg_ParseTuple( args, "O&",
                          PyArray_Converter, &image ))
        goto done;

    npy_intp* dims    = PyArray_DIMS   (image);
    npy_intp* strides = PyArray_STRIDES(image);
    int       ndims   = PyArray_NDIM   (image);
    if( ndims != 2 )
    {
        PyErr_Format(PyExc_RuntimeError, "The input image array must have exactly 2 dims; got %d",
                     ndims);
        goto done;
    }
    if( PyArray_TYPE(image) != NPY_UINT8 )
    {
        PyErr_SetString(PyExc_RuntimeError, "The input image array must contain 8-bit unsigned data");
        goto done;
    }
    if( strides[ndims-1] != 1 )
    {
        PyErr_SetString(PyExc_RuntimeError, "Image rows must live in contiguous memory");
        goto done;
    }


    image_u8_t im = {.width  = dims[1],
                     .height = dims[0],
                     .stride = strides[0],
                     .buf    = PyArray_DATA(image)};

    zarray_t *detections = NULL;  // Declare detections variable outside the GIL macro block
    Py_BEGIN_ALLOW_THREADS  // Release the GIL to allow other Python threads to run
        PyThread_acquire_lock(self->det_lock, 1);  // Acquire the detection lock before running the detector (blocks until the lock is available)
        detections = apriltag_detector_detect(self->td, &im);  // Run detection
        PyThread_release_lock(self->det_lock);  // Release the detection lock
    Py_END_ALLOW_THREADS  // Acquire the GIL after releasing the detection lock

    int N = zarray_size(detections);

    if (N == 0 && errno == EAGAIN){
        PyErr_Format(PyExc_RuntimeError, "Unable to create %d threads for detector", self->td->nthreads);
        goto done;
    }

    detections_tuple = PyTuple_New(N);
    if(detections_tuple == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "Error creating output tuple of size %d", N);
        goto done;
    }

    for (int i=0; i < N; i++)
    {
        xy_c = (PyArrayObject*)PyArray_SimpleNew(1, ((npy_intp[]){2}), NPY_FLOAT64);
        if(xy_c == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError, "Could not allocate xy_c array");
            goto done;
        }
        xy_lb_rb_rt_lt = (PyArrayObject*)PyArray_SimpleNew(2, ((npy_intp[]){4,2}), NPY_FLOAT64);
        if(xy_lb_rb_rt_lt == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError, "Could not allocate xy_lb_rb_rt_lt array");
            goto done;
        }

        apriltag_detection_t* det;
        zarray_get(detections, i, &det);

        *(double*)PyArray_GETPTR1(xy_c, 0) = det->c[0];
        *(double*)PyArray_GETPTR1(xy_c, 1) = det->c[1];

        for(int j=0; j<4; j++)
        {
            *(double*)PyArray_GETPTR2(xy_lb_rb_rt_lt, j, 0) = det->p[j][0];
            *(double*)PyArray_GETPTR2(xy_lb_rb_rt_lt, j, 1) = det->p[j][1];
        }

        // Copy homography matrix for pose estimation
        PyArrayObject* H_arr = (PyArrayObject*)PyArray_SimpleNew(2, ((npy_intp[]){3,3}), NPY_FLOAT64);
        if(H_arr == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError, "Could not allocate H array");
            goto done;
        }
        for(int r=0; r<3; r++)
        {
            for(int c=0; c<3; c++)
            {
                *(double*)PyArray_GETPTR2(H_arr, r, c) = MATD_EL(det->H, r, c);
            }
        }

        PyTuple_SET_ITEM(detections_tuple, i,
                         Py_BuildValue("{s:i,s:f,s:i,s:N,s:N,s:N}",
                                       "hamming", det->hamming,
                                       "margin",  det->decision_margin,
                                       "id",      det->id,
                                       "center",  xy_c,
                                       "lb-rb-rt-lt", xy_lb_rb_rt_lt,
                                       "H", H_arr));
        xy_c           = NULL;
        xy_lb_rb_rt_lt = NULL;
    }
    apriltag_detections_destroy(detections);

    result = detections_tuple;
    detections_tuple = NULL;

  done:
    Py_XDECREF(xy_c);
    Py_XDECREF(xy_lb_rb_rt_lt);
    Py_XDECREF(image);
    Py_XDECREF(detections_tuple);

#ifdef _POSIX_C_SOURCE
    RESET_SIGINT();
#endif
    return result;
}


static const char apriltag_estimate_tag_pose_docstring[] =
    "estimate_tag_pose(detection, info)\n"
    "\n"
    "Estimate the 3D pose of a detected tag using the native AprilTag algorithm.\n"
    "\n"
    "This method uses homography decomposition followed by orthogonal iteration\n"
    "for accurate pose estimation, and handles pose ambiguity by returning the\n"
    "pose with lower object-space error.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "detection : dict\n"
    "    A single detection dictionary from detect().\n"
    "info : dict\n"
    "    Detection info parameters:\n"
    "    - 'tagsize' : float - Physical size of the tag in meters.\n"
    "    - 'fx' : float - Focal length in pixels (x).\n"
    "    - 'fy' : float - Focal length in pixels (y).\n"
    "    - 'cx' : float - Principal point in pixels (x).\n"
    "    - 'cy' : float - Principal point in pixels (y).\n"
    "\n"
    "Returns\n"
    "-------\n"
    "dict with keys:\n"
    "    'R' : numpy.ndarray, shape (3, 3), dtype float64\n"
    "        Rotation matrix from tag frame to camera frame.\n"
    "    't' : numpy.ndarray, shape (3,), dtype float64\n"
    "        Translation vector [x, y, z] in meters (tag origin in camera frame).\n"
    "    'e' : float\n"
    "        Object-space error of the pose estimate.\n"
    "\n"
    "Example\n"
    "-------\n"
    ">>> detector = apriltag('tag36h11')\n"
    ">>> detections = detector.detect(image)\n"
    ">>> info = {'tagsize': 0.1, 'fx': 800, 'fy': 800, 'cx': 320, 'cy': 240}\n"
    ">>> for det in detections:\n"
    "...     pose = detector.estimate_tag_pose(det, info)\n"
    "...     print(f\"Tag {det['id']} at distance {pose['t'][2]:.2f}m, error={pose['e']}\")\n";

static PyObject* apriltag_estimate_tag_pose(apriltag_py_t* self,
                                            PyObject* args,
                                            PyObject* kwargs)
{
    (void)self;  // Unused, pose estimation doesn't need detector state
    PyObject* result = NULL;
    PyObject* detection = NULL;
    PyObject* info_dict = NULL;
    matd_t* H = NULL;

    static char* keywords[] = {"detection", "info", NULL};

    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "OO",
                                    keywords,
                                    &detection,
                                    &info_dict))
    {
        return NULL;
    }

    // Parse info dict
    PyObject* tagsize_obj = PyDict_GetItemString(info_dict, "tagsize");
    PyObject* fx_obj = PyDict_GetItemString(info_dict, "fx");
    PyObject* fy_obj = PyDict_GetItemString(info_dict, "fy");
    PyObject* cx_obj = PyDict_GetItemString(info_dict, "cx");
    PyObject* cy_obj = PyDict_GetItemString(info_dict, "cy");

    if(!tagsize_obj || !fx_obj || !fy_obj || !cx_obj || !cy_obj)
    {
        PyErr_SetString(PyExc_KeyError, "info dict must contain 'tagsize', 'fx', 'fy', 'cx', 'cy'");
        return NULL;
    }

    double tag_size = PyFloat_AsDouble(tagsize_obj);
    double fx = PyFloat_AsDouble(fx_obj);
    double fy = PyFloat_AsDouble(fy_obj);
    double cx = PyFloat_AsDouble(cx_obj);
    double cy = PyFloat_AsDouble(cy_obj);

    if(PyErr_Occurred())
    {
        return NULL;
    }

    // Get corner points from detection dict
    PyObject* corners_obj = PyDict_GetItemString(detection, "lb-rb-rt-lt");
    if(corners_obj == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "detection dict must contain 'lb-rb-rt-lt' key");
        return NULL;
    }

    PyArrayObject* corners = (PyArrayObject*)PyArray_FROM_OTF(corners_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if(corners == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "'lb-rb-rt-lt' must be convertible to float64 array");
        return NULL;
    }

    if(PyArray_NDIM(corners) != 2 || PyArray_DIM(corners, 0) != 4 || PyArray_DIM(corners, 1) != 2)
    {
        Py_DECREF(corners);
        PyErr_SetString(PyExc_ValueError, "'lb-rb-rt-lt' must have shape (4, 2)");
        return NULL;
    }

    // Get homography matrix from detection dict
    PyObject* H_obj = PyDict_GetItemString(detection, "H");
    if(H_obj == NULL)
    {
        Py_DECREF(corners);
        PyErr_SetString(PyExc_KeyError, "detection dict must contain 'H' key (homography matrix)");
        return NULL;
    }

    PyArrayObject* H_arr = (PyArrayObject*)PyArray_FROM_OTF(H_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if(H_arr == NULL)
    {
        Py_DECREF(corners);
        PyErr_SetString(PyExc_TypeError, "'H' must be convertible to float64 array");
        return NULL;
    }

    if(PyArray_NDIM(H_arr) != 2 || PyArray_DIM(H_arr, 0) != 3 || PyArray_DIM(H_arr, 1) != 3)
    {
        Py_DECREF(corners);
        Py_DECREF(H_arr);
        PyErr_SetString(PyExc_ValueError, "'H' must have shape (3, 3)");
        return NULL;
    }

    // Build apriltag_detection_t with corner data and homography
    apriltag_detection_t det;
    memset(&det, 0, sizeof(det));

    double* corner_data = (double*)PyArray_DATA(corners);
    for(int i = 0; i < 4; i++)
    {
        det.p[i][0] = corner_data[i * 2];
        det.p[i][1] = corner_data[i * 2 + 1];
    }

    // Copy homography matrix to matd_t
    H = matd_create(3, 3);
    double* H_data = (double*)PyArray_DATA(H_arr);
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            MATD_EL(H, i, j) = H_data[i * 3 + j];
        }
    }
    det.H = H;

    Py_DECREF(H_arr);

    // Get center if available
    PyObject* center_obj = PyDict_GetItemString(detection, "center");
    if(center_obj != NULL)
    {
        PyArrayObject* center = (PyArrayObject*)PyArray_FROM_OTF(center_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
        if(center != NULL && PyArray_NDIM(center) == 1 && PyArray_DIM(center, 0) == 2)
        {
            double* center_data = (double*)PyArray_DATA(center);
            det.c[0] = center_data[0];
            det.c[1] = center_data[1];
            Py_DECREF(center);
        }
    }

    Py_DECREF(corners);

    // Set up detection info
    apriltag_detection_info_t info;
    info.det = &det;
    info.tagsize = tag_size;
    info.fx = fx;
    info.fy = fy;
    info.cx = cx;
    info.cy = cy;

    // Estimate pose
    apriltag_pose_t pose;
    double err = estimate_tag_pose(&info, &pose);

    // Clean up homography
    matd_destroy(H);

    // Create output arrays
    PyArrayObject* R_arr = (PyArrayObject*)PyArray_SimpleNew(2, ((npy_intp[]){3, 3}), NPY_FLOAT64);
    if(R_arr == NULL)
    {
        matd_destroy(pose.R);
        matd_destroy(pose.t);
        PyErr_SetString(PyExc_RuntimeError, "Could not allocate R array");
        return NULL;
    }

    PyArrayObject* t_arr = (PyArrayObject*)PyArray_SimpleNew(1, ((npy_intp[]){3}), NPY_FLOAT64);
    if(t_arr == NULL)
    {
        Py_DECREF(R_arr);
        matd_destroy(pose.R);
        matd_destroy(pose.t);
        PyErr_SetString(PyExc_RuntimeError, "Could not allocate t array");
        return NULL;
    }

    // Copy rotation matrix
    double* R_data = (double*)PyArray_DATA(R_arr);
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            R_data[i * 3 + j] = MATD_EL(pose.R, i, j);
        }
    }

    // Copy translation vector
    double* t_data = (double*)PyArray_DATA(t_arr);
    for(int i = 0; i < 3; i++)
    {
        t_data[i] = MATD_EL(pose.t, i, 0);
    }

    // Clean up matd structures
    matd_destroy(pose.R);
    matd_destroy(pose.t);

    // Build result dict
    result = Py_BuildValue("{s:N,s:N,s:d}",
                           "R", R_arr,
                           "t", t_arr,
                           "e", err);

    return result;
}

#include "apriltag_detect_docstring.h"
#include "apriltag_py_type_docstring.h"

static PyMethodDef apriltag_methods[] =
    { PYMETHODDEF_ENTRY(apriltag_, detect, METH_VARARGS),
      {"estimate_tag_pose", (PyCFunction)(void(*)(void))apriltag_estimate_tag_pose, METH_VARARGS | METH_KEYWORDS, apriltag_estimate_tag_pose_docstring},
      {NULL, NULL, 0, NULL}
    };

static PyTypeObject apriltagType =
{
     PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "apriltag",
    .tp_basicsize = sizeof(apriltag_py_t),
    .tp_new       = apriltag_new,
    .tp_dealloc   = (destructor)apriltag_dealloc,
    .tp_methods   = apriltag_methods,
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_doc       = apriltag_py_type_docstring
};

static PyMethodDef methods[] =
    { {NULL, NULL, 0, NULL}
    };


#if PY_MAJOR_VERSION == 2

PyMODINIT_FUNC initapriltag(void)
{
    if (PyType_Ready(&apriltagType) < 0)
        return;

    PyObject* module = Py_InitModule3("apriltag", methods,
                                      "AprilTags visual fiducial system detector");

    Py_INCREF(&apriltagType);
    PyModule_AddObject(module, "apriltag", (PyObject *)&apriltagType);

    import_array();
}

#else

static struct PyModuleDef module_def =
    {
     PyModuleDef_HEAD_INIT,
     "apriltag",
     "AprilTags visual fiducial system detector",
     -1,
     methods,
    0,
    0,
    0,
    0
    };

PyMODINIT_FUNC PyInit_apriltag(void)
{
    if (PyType_Ready(&apriltagType) < 0)
        return NULL;

    PyObject* module =
        PyModule_Create(&module_def);

    Py_INCREF(&apriltagType);
    PyModule_AddObject(module, "apriltag", (PyObject *)&apriltagType);

    import_array();

    return module;
}

#endif
