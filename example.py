import cv2
import numpy as np
from apriltag import apriltag
import pathlib

image_dir = pathlib.Path("test/data")

for image_path in image_dir.glob("*.jpg"):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    detector = apriltag("tag36h11")

    detections = detector.detect(image)
    # Visualize results
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for detection in detections:
        corners = detection["lb-rb-rt-lt"].astype(int)
        for i in range(4):
            cv2.line(
                image_color,
                tuple(corners[i]),
                tuple(corners[(i + 1) % 4]),
                (0, 255, 0),
                2,
            )
        center = tuple(detection["center"].astype(int))
        cv2.circle(image_color, center, 5, (0, 0, 255), -1)
        cv2.putText(
            image_color,
            str(detection["id"]),
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )
    cv2.imshow("Detections", image_color)
    cv2.waitKey(0)
cv2.destroyAllWindows()
