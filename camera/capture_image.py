import cv2

def capture_image(save_path="captured_eye.jpg"):
    cap = cv2.VideoCapture(0)
    print("Press SPACE to capture image")

    while True:
        ret, frame = cap.read()
        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1)
        if key == 32:  # SPACE
            cv2.imwrite(save_path, frame)
            break

    cap.release()
    cv2.destroyAllWindows()
    return save_path
