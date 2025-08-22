import cv2

def lazy_frame_generator(source_path, start=0, end=None):
    cap = cv2.VideoCapture(source_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or (end is not None and frame_count >= end):
            break
        if frame_count >= start:
            yield frame
        frame_count += 1

    cap.release()
