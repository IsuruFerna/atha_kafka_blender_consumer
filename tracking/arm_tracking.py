from cvzone.PoseModule import PoseDetector
import cv2 as cv

# Initialize the webcam (try index 0 first, then adjust if needed)
video = cv.VideoCapture(0)

def armDetectFunction(video, quit_key):
    # Check if the webcam opened successfully
    if not video.isOpened():
        print("Error: Could not open webcam at index 2. Try a different index (e.g., 0 or 1).")
        exit()

    # Initialize the PoseDetector class with the given parameters
    detector = PoseDetector(staticMode=False,
                            modelComplexity=1,
                            smoothLandmarks=True,
                            enableSegmentation=False,
                            smoothSegmentation=True,
                            detectionCon=0.5,
                            trackCon=0.5)

    # Loop to continuously get frames from the webcam
    while True:
        # Capture each frame from the webcam
        success, img = video.read()

        # Check if frame was captured successfully
        if not success or img is None:
            print("Error: Failed to capture frame. Check your webcam.")
            break

        # Find the human pose in the frame
        img = detector.findPose(img)

        # Find the landmarks, bounding box, and center of the body in the frame
        lmList, bboxInfo = detector.findPosition(img, draw=True, bboxWithHands=False)

        # Check if any body landmarks are detected
        if lmList:
            # Get the center of the bounding box around the body
            center = bboxInfo["center"]

            # Draw a circle at the center of the bounding box
            cv.circle(img, center, 5, (255, 0, 255), cv.FILLED)

            # Calculate the distance between landmarks 11 and 15 and draw it
            length, img, info = detector.findDistance(lmList[11][0:2],
                                                    lmList[15][0:2],
                                                    img=img,
                                                    color=(255, 0, 0),
                                                    scale=10)

            # Calculate the angle between landmarks 11, 13, and 15 and draw it
            angle, img = detector.findAngle(lmList[11][0:2],
                                            lmList[13][0:2],
                                            lmList[15][0:2],
                                            img=img,
                                            color=(0, 0, 255),
                                            scale=10)

            # Check if the angle is close to 50 degrees with an offset of 10
            isCloseAngle50 = detector.angleCheck(myAngle=angle,
                                                targetAngle=50,
                                                offset=10)

            # Print the result of the angle check
            print(isCloseAngle50)

        # Display the frame in a window
        cv.imshow("Image", img)

        # Wait for 1 millisecond; exit on 'q' key press
        if cv.waitKey(1) & 0xFF == ord(quit_key):
            break

    # Release the webcam and close all windows
    video.release()
    cv.destroyAllWindows()

# armDetectFunction(video, 'k')