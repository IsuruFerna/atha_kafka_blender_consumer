import sys
sys.path.append(r"C:\Users\isuru\Desktop\atha\blender_venv\Lib\site-packages")

import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
from cvzone.PoseModule import PoseDetector
import mediapipe as mp
import numpy as np
import math
import time
from kafka_producer import producer, produce_message



# calculate fingure angles
# TODO: handle value error
def calculate_finger_flexion(base, joint, tip, is_thumb=False):
    """
    Calculate finger flexion angle around X-axis (0° = extended, 90° = fully flexed).
    
    Args:
        base: 3D point [x, y, z] (finger base)
        joint: 3D point [x, y, z] (middle joint)
        tip: 3D point [x, y, z] (fingertip)
        is_thumb: Boolean, adjust for thumb's unique orientation
    
    Returns:
        float: Flexion angle in degrees (0 to 90)
    """
    # Convert to numpy arrays
    p1 = np.array(base, dtype=np.float32)
    p2 = np.array(joint, dtype=np.float32)
    p3 = np.array(tip, dtype=np.float32)
    
    # Calculate vectors
    vec_base_to_joint = p2 - p1  # Proximal segment
    vec_joint_to_tip = p3 - p2   # Distal segment
    
    # For thumb, use XZ plane (sideways bend); for others, use YZ plane (downward bend)
    if is_thumb:
        v1 = vec_base_to_joint[[0, 2]]  # [x, z]
        v2 = vec_joint_to_tip[[0, 2]]   # [x, z]
    else:
        v1 = vec_base_to_joint[1:]  # [y, z]
        v2 = vec_joint_to_tip[1:]   # [y, z]
    
    # Calculate magnitudes
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    
    if mag1 == 0 or mag2 == 0:
        return 0

    # Dot product
    dot_product = np.dot(v1, v2)
    
    # Calculate angle in radians
    cos_angle = np.clip(dot_product / (mag1 * mag2), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    # Determine sign (positive for flextion)
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    if cross < 0:
        angle_deg = -angle_deg
    
    # Determine sign (positive for downward bend)
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    if cross < 0:
        angle_deg = -angle_deg
    
    # Normalize to 0-90° (0 = extended, 90 = flexed)
    # For fingers: positive angle (downward) = flexion
    # For thumb: adjust based on sideways bend
    if is_thumb:
        # Thumb flexes sideways; map to 0-90 based on orientation
        angle_deg = abs(angle_deg)  # Thumb flexion can be positive/negative
    else:
        # Other fingers: downward bend is positive
        angle_deg = max(0, angle_deg)  # Clip negative (extension beyond straight)
    
    return min(angle_deg, 90)  # Cap at 90°

def get_finger_angles(landmarks):
    """
    Calculate flexion angles for all fingers from MediaPipe landmarks.
    
    Args:
        landmarks: MediaPipe hand landmarks
    
    Returns:
        dict: Flexion angles for each finger (0-90°)
    """
    # Landmark indices for fingers (base, proximal joint, tip)
    finger_points = {
        "thumb": [2, 3, 4], 
        "index": [5, 6, 8],
        "middle": [9, 10, 12],
        "ring": [13, 14, 16],
        "pinkey": [17, 18, 20]
    }

    angles = {}
    for finger, indices in finger_points.items():
        base = [landmarks[indices[0]].x, landmarks[indices[0]].y, landmarks[indices[0]].z]
        joint = [landmarks[indices[1]].x, landmarks[indices[1]].y, landmarks[indices[1]].z]
        tip = [landmarks[indices[2]].x, landmarks[indices[2]].y, landmarks[indices[2]].z]

        is_thumb = (finger == "thumb")
        angle = calculate_finger_flexion(base, joint, tip, is_thumb)
        angles[finger] = angle

    return 

# function to compute normal vector from wrist, index base, and pinkey base
def compute_normal(lmList):
    """Calculate the normal vector of the plane defined by wrist (0), index base (5), and pinkey base (17)."""
    p0 = np.array(lmList[0])
    p5 = np.array(lmList[5])
    p17 = np.array(lmList[17])
    vec_a = p5 - p0
    vec_b = p17 - p0
    normal = np.cross(vec_a, vec_b)
    norm = np.linalg.norm(normal)
    return normal / norm if norm != 0 else normal

# Function to compute rotation metrix
def rotate_matrix_from_normals(n1, n2):
    v = np.cross(n1, n2)
    c = np.dot(n1, n2)
    if c < -0.999: # 180-degree flip
        return -np.eye(3)
    s = np.linalg.norm(v)
    if s < 1e-6: # no rotation
        return np.eye(3)
    k = v / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 -c) * np.dot(K, K)

# Function to extract Euler angles
def euler_angles_from_rotation_matrix(R):
    if abs(R[2, 0]) < 0.999:
        theta_y = -np.arcsin(R[2, 0])  # Pitch (Y)
        theta_x = np.arctan2(R[2, 1] / np.cos(theta_y), R[2, 2] / np.cos(theta_y))  # Roll (X)
        theta_z = np.arctan2(R[1, 0] / np.cos(theta_y), R[0, 0] / np.cos(theta_y))  # Yaw (Z)
    else:  # Gimbal lock
        theta_y = np.pi / 2 if R[2, 0] > 0 else -np.pi / 2
        theta_x = 0
        theta_z = np.arctan2(-R[0, 1], -R[0, 2] if theta_y > 0 else R[0, 2])
    return np.degrees(theta_x), np.degrees(theta_y), np.degrees(theta_z)


# initialize MediaPipe Hands
mp_hands = mp.solutions.hands  # Access the hands module
hands = mp_hands.Hands(        # Create Hands object
    max_num_hands=2,
    min_detection_confidence=0.7
)
mp_drawings = mp.solutions.drawing_utils


# right hand reccommended
# Initialize hand detector
hand_detector = HandDetector(detectionCon=0.8, maxHands=1)
# Get reference normal from the first detected frame
ref_normal = None

# Initialize the PoseDetector class with the given parameters
pose_detector = PoseDetector(
    staticMode=False,
    modelComplexity=1,
    smoothLandmarks=True,
    enableSegmentation=False,
    smoothSegmentation=True,
    detectionCon=0.5,
    trackCon=0.5
)

global detected_angles
detected_angles = {
    "hand_R": {
        "thumb": 0,
        "index": 0,
        "middle": 0,
        "ring": 0,
        "pinkey": 0,
    },
    "arm_R": {
        "sholder": 0,
        "elbow": 0,
        "wrist": 0,
    }
}

video = cv.VideoCapture(0)

if not video.isOpened():
    print("Error: Could not open webcam at index 2. Try a different index (e.g., 0 or 1)")
    exit()

while True:
    succsess, frame = video.read()
    
    if not succsess:
        print("Error: Failed to capure frame")
        break

    # flip camera view
    frame = cv.flip(frame, 1)
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # get handedness (Left or Right)
            handedness = results.multi_handedness[idx].classification[0].label

            # Process only the right hand
            if handedness == "Right":
                mp_drawings.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                angles = get_finger_angles(hand_landmarks.landmark)
                
                # Display angles
                y_pos = 30
                for finger, angle in angles.items():
                    text = f"{finger.capitalize()}: {angle:.1f}°"
                    cv.putText(frame, text, (10, y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_pos += 30

                # Label the hand
                cv.putText(frame, "Right Hand", (10, y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # detect hand
    # hands_, hand_img = hand_detector.findHands(frame)  
    
    # # Find the human pose in the frame (using the same frame as hand detection)
    # pose_img = pose_detector.findPose(hand_img) 
    # lmList_pose, bboxInfo = pose_detector.findPosition(pose_img, draw=True, bboxWithHands=False)


    # if hands:
    #     print("hand found")
    #     lmlist = hands[0] # Lista dei punti di riferimento della mano
    #     fingerUp = hand_detector.fingersUp(lmlist)

    #     # print("this is limit: ", type(lmlist), lmlist['lmList'])
    #     arr_lmlist = lmlist['lmList']


    #     if ref_normal is None:
    #         ref_normal = compute_normal(arr_lmlist)
    #         print("Reference normal set.")

    #     curr_normal = compute_normal(arr_lmlist)

    #     # Calculate rotation
    #     R = rotate_matrix_from_normals(ref_normal, curr_normal)
    #     angles = euler_angles_from_rotation_matrix(R)

    #     # Display angles on frame
    #     text = f"X: {angles[0]:.2f}, Y: {angles[1]:.2f}, Z: {angles[2]:.2f}"
    #     cv.putText(frame, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    #     # Calcola gli angoli per ogni dito (esempio per il pollice, indice, medio, anulare, mignolo)
    #     if len(arr_lmlist) > 20:  # Controlla che ci siano abbastanza punti di riferimento
    #         # Angolo per il pollice
    #         # angle_thumb = calculate_angle(arr_lmlist[2], arr_lmlist[3], arr_lmlist[4])

    #         angle_thumb = calculate_finger_flexion(arr_lmlist[2], arr_lmlist[3], arr_lmlist[4])
    #         # print("Angolo pollice: ", angle_thumb)

    #         # Angolo per l'indice
    #         # tofix: UnboundLocalError: cannot access local variable 'angle' where it is not associated with a value
    #         angle_index = calculate_finger_flexion(arr_lmlist[5], arr_lmlist[6], arr_lmlist[7])
    #         # print("Angolo indice: ", angle_index)

    #         # Angolo per il medio
    #         angle_middle = calculate_finger_flexion(arr_lmlist[9], arr_lmlist[10], arr_lmlist[11])
    #         # print("Angolo medio: ", angle_middle)

    #         # Angolo per l'anulare
    #         angle_ring = calculate_finger_flexion(arr_lmlist[13], arr_lmlist[14], arr_lmlist[15])
    #         # print("Angolo anulare: ", angle_ring)
    
    #         # Angolo per il mignolo
    #         angle_pinky = calculate_finger_flexion(arr_lmlist[17], arr_lmlist[18], arr_lmlist[19])
    #         # print("Angolo mignolo: ", angle_pinkey)

    #         # print("finger angles: ", angle_thumb, angle_index, angle_middle, angle_ring, angle_pinky)

    #         detected_angles["hand_R"]["thumb"] = angle_thumb if angle_thumb > 90 else angle_thumb * 3
    #         detected_angles["hand_R"]["index"] = angle_index
    #         detected_angles["hand_R"]["middle"] = angle_middle
    #         detected_angles["hand_R"]["ring"] = angle_ring
    #         detected_angles["hand_R"]["pinkey"] = angle_pinky

    #         print(f"finger rotation: [{angle_thumb}, {angle_index}, {angle_middle}, {angle_ring}, {angle_pinky}]")
    #         # time.sleep(1)

    #         # print(f"hand rot: {arr_lmlist[5]}, {arr_lmlist[9]}, {arr_lmlist[13]}, {arr_lmlist[17]}")

    # if lmList_pose:
    #     # # Get the center of the bounding box around the body
    #     # center = bboxInfo["center"]

    #     # # Draw a circle at the center of the bounding box
    #     # cv.circle(pose_img, center, 5, (255, 0, 255), cv.FILLED)

    #     # # Example: Highlight wrist (landmark 15 or 16) from pose
    #     # wrist_idx = 15  # Left wrist (adjust to 16 for right if needed)
    #     # x, y = lmList_pose[wrist_idx][0:2]
    #     # cv.circle(pose_img, (x, y), 5, (255, 0, 0), cv.FILLED)
    #     # cv.putText(pose_img, "Pose Wrist", (x + 10, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    #     sholder_r_angle, img = pose_detector.findAngle(
    #             lmList_pose[12][0:2],
    #             lmList_pose[11][0:2],
    #             lmList_pose[13][0:2],
    #             img=pose_img,
    #             color=(0, 255, 0),
    #             scale=10
    #         )
        
    #     elbow_r_angle, img = pose_detector.findAngle(
    #             lmList_pose[11][0:2],
    #             lmList_pose[13][0:2],
    #             lmList_pose[15][0:2],
    #             img=pose_img,
    #             color=(0, 255, 0),
    #             scale=10
    #         )
        
    #     wrist_r_andgle, img = pose_detector.findAngle(
    #             lmList_pose[13][0:2],
    #             lmList_pose[15][0:2],
    #             lmList_pose[19][0:2],
    #             img=pose_img,
    #             color=(0, 255, 0),
    #             scale=10
    #         )
        
    #     # FIXME: correct angles
    #     # print("arm rototions: ", sholder_r_angle, elbow_r_angle, wrist_r_andgle)
        
    #     detected_angles["arm_R"]["sholder"] = sholder_r_angle
    #     detected_angles["arm_R"]["elbow"] = elbow_r_angle
    #     detected_angles["arm_R"]["wrist"] = wrist_r_andgle
        
    #     produce_message(detected_angles)
    #     time.sleep(0.1)
    #     producer.flush()
            
    # Show combined frame
    # cv.imshow("Hand and Pose Tracking with Rotation", pose_img)

    # k = cv.waitKey(1)
    # if k == ord('q'):
    #     break
    if cv.waitKey(1) & 0xFF == ord('q'):
        # detected_angles["hand_R"]["thumb"] = 0
        # detected_angles["hand_R"]["index"] = 0
        # detected_angles["hand_R"]["middle"] = 0
        # detected_angles["hand_R"]["ring"] = 0
        # detected_angles["hand_R"]["pinkey"] = 0

        # detected_angles["arm_R"]["sholder"] = 0
        # detected_angles["arm_R"]["elbow"] = 0
        # detected_angles["arm_R"]["wrist"] = 0
        break

video.release()
cv.destroyAllWindows()
hands.close()


# handDetectFunction(video, 'k')