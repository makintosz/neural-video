import os
import sys

import cv2.cv2 as cv2

video_file_name = sys.argv[1]
video_capture = cv2.VideoCapture(video_file_name)
success, frame = video_capture.read()
frame_counter = 0
while success:
    cv2.imwrite(os.path.join(sys.argv[2], f"{frame_counter}.jpg"), frame)
    frame_counter += 1
    success, frame = video_capture.read()
