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
# Example 4: Pose Estimation (requires camera calibration)
# =============================================================================
def estimate_pose(detection, camera_matrix, tag_size):
    """
    Estimate 3D pose of a detected tag.

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

    # Solve PnP
    dist_coeffs = np.zeros(4)  # Assuming no distortion
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
    )

    return rvec, tvec if success else (None, None)


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
  python example.py                     # Run interactive visualization
  python example.py --example basic     # Run basic detection
  python example.py --example config    # Run configured detection
  python example.py --example batch     # Process all images in directory
  python example.py --example families  # Detect multiple tag families
  python example.py --image photo.jpg   # Use specific image
  python example.py --list              # List available examples
        """,
    )
    parser.add_argument(
        "--example",
        "-e",
        choices=["interactive", "basic", "config", "batch", "families"],
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

    args = parser.parse_args()

    if args.list:
        print("""
Available examples:
  interactive  - Visualize detections with OpenCV, navigate with any key, 'q' to quit
  basic        - Minimal detection example, prints results to console
  config       - Detection with custom parameters (threads, decimate, blur)
  batch        - Process all images in a directory
  families     - Detect tags from multiple families in the same image

Tag families:
  tag36h11 (recommended), tag36h10, tag25h9, tag16h5,
  tagCircle21h7, tagCircle49h12, tagStandard41h12,
  tagStandard52h13, tagCustom48h12
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

    elif args.example == "batch":
        results = batch_process(args.image_dir, family=args.family)
        total = sum(len(dets) for dets in results.values())
        print(f"\nTotal: {total} tags in {len(results)} images")

    elif args.example == "families":
        for image_path in image_paths:
            print(f"\n=== {image_path.name} ===")
            detect_multiple_families(image_path)
