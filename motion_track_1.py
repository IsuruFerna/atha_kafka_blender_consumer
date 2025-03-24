import sys
sys.path.append(r"C:\Users\isuru\Desktop\atha\blender_venv\Lib\site-packages")

import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import bpy
import math
import time

# calculate fingure angles
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
    
    return angle_deg

if __name__ == "__main__":
    detector = HandDetector(detectionCon=0.8, maxHands=1)

    video = cv.VideoCapture(0)

    pose_bones = bpy.data.objects["metarig"].pose.bones
    f_index_1_r = pose_bones["f_index.01.R"]
    f_middle_1_r = pose_bones["f_middle.01.R"]
    f_ring_1_r = pose_bones["f_ring.01.R"]
    f_pinky_1_r = pose_bones["f_pinky.01.R"]

    while True:
        ret, frame = video.read()
        frame = cv.flip(frame, 1)
        hands, img = detector.findHands(frame)
        
        if hands:
            lmlist = hands[0] # Lista dei punti di riferimento della mano
            fingerUp = detector.fingersUp(lmlist)

            # print("this is limit: ", type(lmlist), lmlist['lmList'])k
            arr_lmlist = lmlist['lmList']
            
            # Calcola gli angoli per ogni dito (esempio per il pollice, indice, medio, anulare, mignolo)
            if len(arr_lmlist) > 20:  # Controlla che ci siano abbastanza punti di riferimento
                # Angolo per il pollice
                # angle_thumb = calculate_angle(arr_lmlist[2], arr_lmlist[3], arr_lmlist[4])

                angle_thumb = calculate_finger_flexion(arr_lmlist[2], arr_lmlist[3], arr_lmlist[4])
                print("Angolo pollice: ", angle_thumb)

                # Angolo per l'indice
                # tofix: UnboundLocalError: cannot access local variable 'angle' where it is not associated with a value
                angle_index = calculate_finger_flexion(arr_lmlist[5], arr_lmlist[6], arr_lmlist[7])
                # print("Angolo indice: ", angle_index)
                f_index_1_r.rotation_euler[0] = math.radians(angle_index)

                
                # Angolo per il medio
                angle_middle = calculate_finger_flexion(arr_lmlist[9], arr_lmlist[10], arr_lmlist[11])
                # print("Angolo medio: ", angle_middle)
                f_middle_1_r.rotation_euler[0] = math.radians(angle_middle)

                # Angolo per l'anulare
                angle_ring = calculate_finger_flexion(arr_lmlist[13], arr_lmlist[14], arr_lmlist[15])
                # print("Angolo anulare: ", angle_ring)
                f_ring_1_r.rotation_euler[0] = math.radians(angle_ring)

                # Angolo per il mignolo
                angle_pinky = calculate_finger_flexion(arr_lmlist[17], arr_lmlist[18], arr_lmlist[19])
                # print("Angolo mignolo: ", angle_pinky)
                f_pinky_1_r.rotation_euler[0] = math.radians(angle_pinky)

                bpy.context.view_layer.update()
                bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                time.sleep(0.1)
                
        # Mostra il frame con gli angoli e la conta delle dita
        cv.imshow("Hand Tracking", frame)

        k = cv.waitKey(1)
        if k == ord("k"):
            f_index_1_r.rotation_euler[0] = 0
            f_middle_1_r.rotation_euler[0] = 0
            f_ring_1_r.rotation_euler[0] = 0
            f_pinky_1_r.rotation_euler[0] = 0
            break

    video.release()
    cv.destroyAllWindows()