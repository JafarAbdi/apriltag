"""
AprilTag Python Wrapper - Usage Examples

This module demonstrates how to use the apriltag Python bindings for detecting
AprilTag fiducial markers in images.

AprilTag is a visual fiducial system useful for robotics, augmented reality,
and camera calibration applications.

API Reference
=============

apriltag(family, **kwargs)
--------------------------
Create an AprilTag detector instance.

Parameters:
    family : str (required)
        Tag family to detect. Supported families:
        - "tag36h11"       - Recommended. 36-bit tags, 11-bit hamming code
        - "tag36h10"       - 36-bit tags, 10-bit hamming code
        - "tag25h9"        - 25-bit tags, 9-bit hamming code (smaller markers)
        - "tag16h5"        - 16-bit tags, 5-bit hamming code (smallest markers)
        - "tagCircle21h7"  - Circular tags, 21-bit, 7-bit hamming
        - "tagCircle49h12" - Circular tags, 49-bit, 12-bit hamming
        - "tagStandard41h12" - Standard 41-bit, 12-bit hamming
        - "tagStandard52h13" - Standard 52-bit, 13-bit hamming
        - "tagCustom48h12" - Custom 48-bit, 12-bit hamming

    threads : int, default=1
        Number of CPU threads to use for detection.

    maxhamming : int, default=1
        Maximum hamming distance for error correction (0-3).
        Higher values detect more damaged tags but increase false positives.

    decimate : float, default=2.0
        Decimation factor for quad detection. Higher values speed up
        detection but reduce accuracy for small tags.

    blur : float, default=0.0
        Gaussian blur sigma applied before edge detection.
        Can help with noisy images.

    refine_edges : bool, default=True
        Whether to refine tag edges for sub-pixel accuracy.

    debug : bool, default=False
        Enable debug mode (writes debug images to current directory).

detector.detect(image)
----------------------
Detect AprilTags in an image.

Parameters:
    image : numpy.ndarray
        Grayscale image as 2D uint8 numpy array.
        Rows must be contiguous in memory.

Returns:
    tuple of dict
        Each detection is a dictionary with:
        - "id" : int
            The tag ID number.
        - "hamming" : int
            Number of bit errors corrected (0 = perfect detection).
        - "margin" : float
            Decision margin. Higher values indicate more confident detections.
        - "center" : numpy.ndarray, shape (2,), dtype float64
            Tag center coordinates [x, y].
        - "lb-rb-rt-lt" : numpy.ndarray, shape (4, 2), dtype float64
            Corner coordinates in order: left-bottom, right-bottom,
            right-top, left-top. Each row is [x, y].

detector.estimate_tag_pose(detection, info)
-------------------------------------------
Estimate 3D pose of a detected tag using native AprilTag algorithm.

This method uses homography decomposition followed by orthogonal iteration
for accurate pose estimation, and handles pose ambiguity automatically.

Parameters:
    detection : dict
        A single detection dictionary from detect().
    info : dict
        Detection info with keys:
        - "tagsize" : float - Physical tag size in meters.
        - "fx" : float - Focal length x in pixels.
        - "fy" : float - Focal length y in pixels.
        - "cx" : float - Principal point x in pixels.
        - "cy" : float - Principal point y in pixels.

Returns:
    dict with keys:
        - "R" : numpy.ndarray, shape (3, 3), dtype float64
            Rotation matrix from tag frame to camera frame.
        - "t" : numpy.ndarray, shape (3,), dtype float64
            Translation vector [x, y, z] in meters.
        - "e" : float
            Object-space error of the pose estimate.
"""

import cv2
import numpy as np
from apriltag import apriltag
import pathlib


# =============================================================================
# Example 1: Basic Detection
# =============================================================================
def basic_detection(image_path):
    """Minimal example of AprilTag detection."""
    # Load image as grayscale
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    # Create detector with default settings
    detector = apriltag("tag36h11")

    # Detect tags
    detections = detector.detect(image)

    for det in detections:
        print(f"Detected tag ID: {det['id']}")
        print(f"  Center: {det['center']}")
        print(f"  Hamming: {det['hamming']}")
        print(f"  Margin: {det['margin']:.2f}")

    return detections


# =============================================================================
# Example 2: Custom Detector Configuration
# =============================================================================
def configured_detection(image_path):
    """Example with custom detector parameters for performance tuning."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    # High-performance configuration for large images
    detector = apriltag(
        "tag36h11",
        threads=4,  # Use 4 CPU threads
        decimate=4.0,  # Aggressive decimation for speed
        blur=0.8,  # Slight blur to reduce noise
        refine_edges=True,  # Keep edge refinement for accuracy
        maxhamming=2,  # Allow more bit errors
    )

    return detector.detect(image)


# =============================================================================
# Example 3: Visualization with OpenCV
# =============================================================================
def visualize_detections(image_path):
    """Detect and visualize AprilTags with OpenCV."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    detector = apriltag("tag36h11")
    detections = detector.detect(image)

    # Convert to color for visualization
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for det in detections:
        # Draw tag outline
        corners = det["lb-rb-rt-lt"].astype(int)
        for i in range(4):
            pt1 = tuple(corners[i])
            pt2 = tuple(corners[(i + 1) % 4])
            cv2.line(image_color, pt1, pt2, (0, 255, 0), 2)

        # Draw center point
        center = tuple(det["center"].astype(int))
        cv2.circle(image_color, center, 5, (0, 0, 255), -1)

        # Draw tag ID
        cv2.putText(
            image_color,
            f"ID: {det['id']}",
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

    return image_color, detections


# =============================================================================
# Example 4a: Pose Estimation with OpenCV solvePnP
# =============================================================================
def estimate_pose_opencv(detection, camera_matrix, tag_size):
    """
    Estimate 3D pose of a detected tag using OpenCV's solvePnP.

    This is a simple approach that works without the native pose estimation.
    For better accuracy, use detector.estimate_tag_pose() instead.

    Parameters:
        detection : dict
            Single detection from detector.detect()
        camera_matrix : numpy.ndarray, shape (3, 3)
            Camera intrinsic matrix
        tag_size : float
            Physical size of the tag in meters

    Returns:
        rvec : numpy.ndarray
            Rotation vector (Rodrigues format)
        tvec : numpy.ndarray
            Translation vector (tag position in camera frame)
    """
    # Define tag corners in 3D (tag coordinate system, z=0)
    half = tag_size / 2
    object_points = np.array(
        [
            [-half, -half, 0],  # left-bottom
            [half, -half, 0],  # right-bottom
            [half, half, 0],  # right-top
            [-half, half, 0],  # left-top
        ],
        dtype=np.float64,
    )

    # Get detected corners
    image_points = detection["lb-rb-rt-lt"].astype(np.float64)

    # Solve PnP using IPPE_SQUARE (optimized for square planar markers)
    dist_coeffs = np.zeros(4)  # Assuming no distortion
    retval, rvecs, tvecs, reprojection_errors = cv2.solvePnPGeneric(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )

    if retval == 0 or len(rvecs) == 0:
        return (None, None)

    # IPPE_SQUARE returns up to 2 solutions, first one has lower reprojection error
    return (rvecs[0], tvecs[0])


def visualize_poses_opencv(image_path, camera_params, tag_size=0.1):
    """
    Detect tags and visualize poses using OpenCV's solvePnP.

    Parameters:
        image_path : path-like
            Path to input image.
        camera_params : tuple of 4 floats
            Camera intrinsic parameters (fx, fy, cx, cy) in pixels.
        tag_size : float
            Physical size of the tag in meters.

    Returns:
        image_color : numpy.ndarray
            Image with pose axes drawn.
        results : list of dict
            Pose results for each detected tag.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    detector = apriltag("tag36h11")
    detections = detector.detect(image)

    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    fx, fy, cx, cy = camera_params
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float64)

    results = []
    for det in detections:
        rvec, tvec = estimate_pose_opencv(det, camera_matrix, tag_size)
        if rvec is None:
            continue

        results.append({
            "id": det["id"],
            "rvec": rvec,
            "tvec": tvec,
        })

        # Draw tag outline
        corners = det["lb-rb-rt-lt"].astype(int)
        for i in range(4):
            cv2.line(
                image_color,
                tuple(corners[i]),
                tuple(corners[(i + 1) % 4]),
                (0, 255, 0),
                2,
            )

        # Draw 3D coordinate axes
        axis_length = tag_size * 0.5
        cv2.drawFrameAxes(
            image_color,
            camera_matrix,
            np.zeros(4),  # No distortion
            rvec,
            tvec,
            axis_length,
        )

        # Draw distance text
        center = tuple(det["center"].astype(int))
        distance = np.linalg.norm(tvec)
        cv2.putText(
            image_color,
            f"ID:{det['id']} {distance:.2f}m",
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

        print(f"Tag {det['id']} (OpenCV):")
        print(f"  Position: x={tvec[0][0]:.3f}m, y={tvec[1][0]:.3f}m, z={tvec[2][0]:.3f}m")
        print(f"  Distance: {distance:.3f}m")

    return image_color, results


# =============================================================================
# Example 4b: Pose Estimation with Native AprilTag Algorithm
# =============================================================================
def pose_estimation_example(image_path, camera_params, tag_size=0.1):
    """
    Detect tags and estimate their 3D poses using native AprilTag algorithm.

    Parameters:
        image_path : path-like
            Path to input image.
        camera_params : tuple of 4 floats
            Camera intrinsic parameters (fx, fy, cx, cy) in pixels.
        tag_size : float
            Physical size of the tag in meters (default: 0.1m = 10cm).

    Returns:
        list of dict
            Each dict contains detection info and pose:
            - "id": tag ID
            - "R": 3x3 rotation matrix
            - "t": translation vector [x, y, z]
            - "e": pose estimation error
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    detector = apriltag("tag36h11")
    detections = detector.detect(image)

    # Build info dict (mirrors C API's apriltag_detection_info_t)
    fx, fy, cx, cy = camera_params
    info = {
        "tagsize": tag_size,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
    }

    results = []
    for det in detections:
        # Use native pose estimation
        pose = detector.estimate_tag_pose(det, info)

        results.append({
            "id": det["id"],
            "R": pose["R"],
            "t": pose["t"],
            "e": pose["e"],
        })

        # Print pose info
        t = pose["t"]
        distance = np.linalg.norm(t)
        print(f"Tag {det['id']}:")
        print(f"  Position: x={t[0]:.3f}m, y={t[1]:.3f}m, z={t[2]:.3f}m")
        print(f"  Distance: {distance:.3f}m")
        print(f"  Error: {pose['e']:.6f}")

    return results


def visualize_poses(image_path, camera_params, tag_size=0.1):
    """
    Detect tags, estimate poses, and visualize with 3D axes overlay.

    Parameters:
        image_path : path-like
            Path to input image.
        camera_params : tuple of 4 floats
            Camera intrinsic parameters (fx, fy, cx, cy) in pixels.
        tag_size : float
            Physical size of the tag in meters.

    Returns:
        image_color : numpy.ndarray
            Image with pose axes drawn.
        results : list of dict
            Pose results for each detected tag.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    detector = apriltag("tag36h11")
    detections = detector.detect(image)

    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    fx, fy, cx, cy = camera_params
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float64)

    # Build info dict (mirrors C API's apriltag_detection_info_t)
    info = {
        "tagsize": tag_size,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
    }

    results = []
    for det in detections:
        pose = detector.estimate_tag_pose(det, info)
        results.append({"id": det["id"], **pose})

        # Draw tag outline
        corners = det["lb-rb-rt-lt"].astype(int)
        for i in range(4):
            cv2.line(
                image_color,
                tuple(corners[i]),
                tuple(corners[(i + 1) % 4]),
                (0, 255, 0),
                2,
            )

        # Convert rotation matrix to Rodrigues vector for cv2.drawFrameAxes
        rvec, _ = cv2.Rodrigues(pose["R"])
        tvec = pose["t"].reshape(3, 1)

        # Draw 3D coordinate axes
        axis_length = tag_size * 0.5
        cv2.drawFrameAxes(
            image_color,
            camera_matrix,
            np.zeros(4),  # No distortion
            rvec,
            tvec,
            axis_length,
        )

        # Draw distance text
        center = tuple(det["center"].astype(int))
        distance = np.linalg.norm(pose["t"])
        cv2.putText(
            image_color,
            f"ID:{det['id']} {distance:.2f}m",
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

    return image_color, results


# =============================================================================
# Example 5: Processing Multiple Images
# =============================================================================
def batch_process(image_dir, family="tag36h11"):
    """Process all images in a directory."""
    image_dir = pathlib.Path(image_dir)
    detector = apriltag(family)

    results = {}
    for image_path in image_dir.glob("*.jpg"):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            detections = detector.detect(image)
            results[image_path.name] = detections
            print(f"{image_path.name}: {len(detections)} tags detected")

    return results


# =============================================================================
# Example 6: Different Tag Families
# =============================================================================
def detect_multiple_families(image_path):
    """Detect tags from multiple families in the same image."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    families = ["tag36h11", "tag25h9", "tag16h5"]
    all_detections = {}

    for family in families:
        detector = apriltag(family)
        detections = detector.detect(image)
        all_detections[family] = detections
        print(f"{family}: {len(detections)} tags")

    return all_detections


# =============================================================================
# Main: Command-line Interface
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AprilTag detection examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python example.py                        # Run interactive visualization
  python example.py --example basic        # Run basic detection
  python example.py --example config       # Run configured detection
  python example.py --example pose         # Run native pose estimation
  python example.py --example pose-opencv  # Run OpenCV pose estimation
  python example.py --example batch        # Process all images in directory
  python example.py --example families     # Detect multiple tag families
  python example.py --image photo.jpg      # Use specific image
  python example.py --list                 # List available examples
        """,
    )
    parser.add_argument(
        "--example",
        "-e",
        choices=["interactive", "basic", "config", "pose", "pose-opencv", "batch", "families"],
        default="interactive",
        help="Example to run (default: interactive)",
    )
    parser.add_argument(
        "--image",
        "-i",
        type=pathlib.Path,
        help="Path to input image (default: use test/data/*.jpg)",
    )
    parser.add_argument(
        "--image-dir",
        "-d",
        type=pathlib.Path,
        default=pathlib.Path("test/data"),
        help="Directory containing images (default: test/data)",
    )
    parser.add_argument(
        "--family",
        "-f",
        default="tag36h11",
        help="Tag family to use (default: tag36h11)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available examples and exit",
    )
    parser.add_argument(
        "--tag-size",
        "-s",
        type=float,
        default=0.1,
        help="Tag size in meters for pose estimation (default: 0.1)",
    )
    parser.add_argument(
        "--camera",
        "-c",
        type=float,
        nargs=4,
        metavar=("FX", "FY", "CX", "CY"),
        help="Camera intrinsics: fx fy cx cy (default: estimate from image size)",
    )

    args = parser.parse_args()

    if args.list:
        print("""
Available examples:
  interactive  - Visualize detections with OpenCV, navigate with any key, 'q' to quit
  basic        - Minimal detection example, prints results to console
  config       - Detection with custom parameters (threads, decimate, blur)
  pose         - Native AprilTag pose estimation (more accurate)
  pose-opencv  - OpenCV solvePnP pose estimation (simpler, no native dependency)
  batch        - Process all images in a directory
  families     - Detect tags from multiple families in the same image

Tag families:
  tag36h11 (recommended), tag36h10, tag25h9, tag16h5,
  tagCircle21h7, tagCircle49h12, tagStandard41h12,
  tagStandard52h13, tagCustom48h12

Pose estimation options:
  --camera FX FY CX CY  Camera intrinsics (focal length and principal point)
  --tag-size SIZE       Physical tag size in meters (default: 0.1)
        """)
        exit(0)

    # Get image path(s)
    if args.image:
        image_paths = [args.image]
    else:
        image_paths = list(args.image_dir.glob("*.jpg"))
        if not image_paths:
            image_paths = list(args.image_dir.glob("*.png"))
        if not image_paths:
            print(f"No images found in {args.image_dir}")
            exit(1)

    # Run selected example
    if args.example == "interactive":
        for image_path in image_paths:
            image_color, detections = visualize_detections(image_path)

            print(f"\n{image_path.name}: {len(detections)} detections")
            for det in detections:
                print(
                    f"  Tag {det['id']}: margin={det['margin']:.2f}, "
                    f"hamming={det['hamming']}"
                )

            cv2.imshow("AprilTag Detections", image_color)
            key = cv2.waitKey(0)
            if key == ord("q"):
                break
        cv2.destroyAllWindows()

    elif args.example == "basic":
        for image_path in image_paths:
            print(f"\n=== {image_path.name} ===")
            basic_detection(image_path)

    elif args.example == "config":
        for image_path in image_paths:
            print(f"\n=== {image_path.name} ===")
            detections = configured_detection(image_path)
            print(f"Found {len(detections)} tags with configured detector")
            for det in detections:
                print(f"  Tag {det['id']}: margin={det['margin']:.2f}")

    elif args.example == "pose":
        # Get camera parameters (estimate from first image if not provided)
        if args.camera:
            camera_params = tuple(args.camera)
        else:
            # Estimate camera params from image size (rough approximation)
            sample_image = cv2.imread(str(image_paths[0]), cv2.IMREAD_GRAYSCALE)
            h, w = sample_image.shape
            # Assume 60 degree FOV and principal point at center
            fx = fy = w / (2 * np.tan(np.radians(30)))
            cx, cy = w / 2, h / 2
            camera_params = (fx, fy, cx, cy)
            print(f"Using estimated camera params: fx={fx:.1f}, fy={fy:.1f}, "
                  f"cx={cx:.1f}, cy={cy:.1f}")
            print("For accurate results, provide --camera FX FY CX CY\n")

        for image_path in image_paths:
            print(f"\n=== {image_path.name} ===")
            image_color, results = visualize_poses(
                image_path, camera_params, args.tag_size
            )

            if results:
                cv2.imshow("Pose Estimation", image_color)
                key = cv2.waitKey(0)
                if key == ord("q"):
                    break
            else:
                print("No tags detected")

        cv2.destroyAllWindows()

    elif args.example == "pose-opencv":
        # Get camera parameters (estimate from first image if not provided)
        if args.camera:
            camera_params = tuple(args.camera)
        else:
            # Estimate camera params from image size (rough approximation)
            sample_image = cv2.imread(str(image_paths[0]), cv2.IMREAD_GRAYSCALE)
            h, w = sample_image.shape
            # Assume 60 degree FOV and principal point at center
            fx = fy = w / (2 * np.tan(np.radians(30)))
            cx, cy = w / 2, h / 2
            camera_params = (fx, fy, cx, cy)
            print(f"Using estimated camera params: fx={fx:.1f}, fy={fy:.1f}, "
                  f"cx={cx:.1f}, cy={cy:.1f}")
            print("For accurate results, provide --camera FX FY CX CY\n")

        for image_path in image_paths:
            print(f"\n=== {image_path.name} ===")
            image_color, results = visualize_poses_opencv(
                image_path, camera_params, args.tag_size
            )

            if results:
                cv2.imshow("Pose Estimation (OpenCV)", image_color)
                key = cv2.waitKey(0)
                if key == ord("q"):
                    break
            else:
                print("No tags detected")

        cv2.destroyAllWindows()

    elif args.example == "batch":
        results = batch_process(args.image_dir, family=args.family)
        total = sum(len(dets) for dets in results.values())
        print(f"\nTotal: {total} tags in {len(results)} images")

    elif args.example == "families":
        for image_path in image_paths:
            print(f"\n=== {image_path.name} ===")
            detect_multiple_families(image_path)
