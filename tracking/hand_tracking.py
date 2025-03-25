

import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# calculate fingure angles
# TODO: handle value error
def calculate_finger_flexion(base, joint, tip):
    """
    Calculate the flexion angle of a finger around the X-axis (hand-local).
    
    Args:
        base: 3D point [x, y, z] (finger base)
        joint: 3D point [x, y, z] (middle joint)
        tip: 3D point [x, y, z] (fingertip)
    
    Returns:
        float: Flexion angle in degrees (positive for bending downward)
    """
    # Convert to numpy arrays
    p1 = np.array(base, dtype=np.float32)
    p2 = np.array(joint, dtype=np.float32)
    p3 = np.array(tip, dtype=np.float32)
    
    # Calculate vectors
    vec_base_to_joint = p2 - p1  # Proximal segment
    vec_joint_to_tip = p3 - p2   # Distal segment
    
    # Project vectors onto YZ plane (since X-axis is rotation axis)
    v1_yz = vec_base_to_joint[1:]  # [y, z]
    v2_yz = vec_joint_to_tip[1:]   # [y, z]
    
    # Calculate magnitudes
    mag1 = np.linalg.norm(v1_yz)
    mag2 = np.linalg.norm(v2_yz)
    
    if mag1 == 0 or mag2 == 0:
        raise ValueError("Vector projection in YZ plane has zero magnitude")
    
    # Dot product
    dot_product = np.dot(v1_yz, v2_yz)
    
    # Calculate angle in radians
    cos_angle = np.clip(dot_product / (mag1 * mag2), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    
    # Convert to degrees
    angle_deg = np.degrees(angle_rad)
    
    # Determine sign (positive for downward bend)
    # Cross product in YZ plane (scalar since 2D)
    cross = v1_yz[0] * v2_yz[1] - v1_yz[1] * v2_yz[0]
    if cross < 0:
        angle_deg = -angle_deg
    
    if angle_deg < 0:
        return angle_deg * (-1)
    
    return angle_deg


video = cv.VideoCapture(0)

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

# Get reference normal from the first detected frame
ref_normal = None
detector = HandDetector(detectionCon=0.8, maxHands=1)

while True:
    succsess, frame = video.read()

    if not succsess:
        print("Error: Failed to capure frame")
        break

    # flip camera view
    frame = cv.flip(frame, 1)

    # detect hand
    hands, img = detector.findHands(frame)   

    if hands:
        lmlist = hands[0] # Lista dei punti di riferimento della mano
        fingerUp = detector.fingersUp(lmlist)

        # print("this is limit: ", type(lmlist), lmlist['lmList'])k
        arr_lmlist = lmlist['lmList']

        if ref_normal is None:
            ref_normal = compute_normal(arr_lmlist)
            print("Reference normal set.")

        curr_normal = compute_normal(arr_lmlist)

        # Calculate rotation
        R = rotate_matrix_from_normals(ref_normal, curr_normal)
        angles = euler_angles_from_rotation_matrix(R)

        # Display angles on frame
        text = f"X: {angles[0]:.2f}, Y: {angles[1]:.2f}, Z: {angles[2]:.2f}"
        cv.putText(frame, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Calcola gli angoli per ogni dito (esempio per il pollice, indice, medio, anulare, mignolo)
        if len(arr_lmlist) > 20:  # Controlla che ci siano abbastanza punti di riferimento
            # Angolo per il pollice
            # angle_thumb = calculate_angle(arr_lmlist[2], arr_lmlist[3], arr_lmlist[4])

            angle_thumb = calculate_finger_flexion(arr_lmlist[2], arr_lmlist[3], arr_lmlist[4])
            # print("Angolo pollice: ", angle_thumb)

            # Angolo per l'indice
            # tofix: UnboundLocalError: cannot access local variable 'angle' where it is not associated with a value
            angle_index = calculate_finger_flexion(arr_lmlist[5], arr_lmlist[6], arr_lmlist[7])
            # print("Angolo indice: ", angle_index)

            # Angolo per il medio
            angle_middle = calculate_finger_flexion(arr_lmlist[9], arr_lmlist[10], arr_lmlist[11])
            # print("Angolo medio: ", angle_middle)


            # Angolo per l'anulare
            angle_ring = calculate_finger_flexion(arr_lmlist[13], arr_lmlist[14], arr_lmlist[15])
            # print("Angolo anulare: ", angle_ring)
    
            # Angolo per il mignolo
            angle_pinky = calculate_finger_flexion(arr_lmlist[17], arr_lmlist[18], arr_lmlist[19])
            # print("Angolo mignolo: ", angle_pinky

            # print(f"finger rotation: [{angle_thumb}, {angle_index}, {angle_middle}, {angle_ring}, {angle_pinky}]")
            # time.sleep(1)

            # print(f"hand rot: {arr_lmlist[5]}, {arr_lmlist[9]}, {arr_lmlist[13]}, {arr_lmlist[17]}")
            
    # Mostra il frame con gli angoli e la conta delle dita
    cv.imshow("Hand Tracking", frame)

    k = cv.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv.destroyAllWindows()

# handDetectFunction(video, 'k')