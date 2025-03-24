'''
import cv2 as cv
from cvzone.HandTrackingModule import HandDetector


try:
    detector = HandDetector(detectionCon=0.8, maxHands=1)
except TypeError as e:
    print("Error initializing HandDetector:", e)

video = cv.VideoCapture(0)

while True:
    ret, frame = video.read()
    frame = cv.flip(frame,1)
    hands, img = detector.findHands(frame)
    if hands:
        lmlist = hands[0]
        fingerUp = detector.fingersUp(lmlist)

        print(fingerUp)
        if fingerUp == [0,0,0,0,0]:
            cv.putText(frame,"Finger count:0",(20,460),cv.FONT_HERSHEY_COMPLEX,1,(255,255,255),1,cv.LINE_AA)
        elif fingerUp == [0,1,0,0,0]:
            cv.putText(frame,"Finger count:1",(20,460),cv.FONT_HERSHEY_COMPLEX,1,(255,255,255),1,cv.LINE_AA)
        elif fingerUp == [0,1,1,0,0]:
            cv.putText(frame,"Finger count:2",(20,460),cv.FONT_HERSHEY_COMPLEX,1,(255,255,255),1,cv.LINE_AA)
        elif fingerUp == [0,1,1,1,0]:
            cv.putText(frame,"Finger count:3",(20,460),cv.FONT_HERSHEY_COMPLEX,1,(255,255,255),1,cv.LINE_AA)
        elif fingerUp == [0,1,1,1,1]:
            cv.putText(frame,"Finger count:4",(20,460),cv.FONT_HERSHEY_COMPLEX,1,(255,255,255),1,cv.LINE_AA)
        elif fingerUp == [1,1,1,1,1]:
            cv.putText(frame,"Finger count:5",(20,460),cv.FONT_HERSHEY_COMPLEX,1,(255,255,255),1,cv.LINE_AA)
        
    cv.imshow("frame",frame)
    k = cv.waitKey(1)
    if k == ord("k"):
        break

video.release()
cv.destroyAllWindows'
'''
import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np

# # Funzione per calcolare l'angolo tra tre punti


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


# Inizializza il rilevatore di mani
detector = HandDetector(detectionCon=0.8, maxHands=1)

video = cv.VideoCapture(0)

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
            
            # Angolo per il medio
            angle_middle = calculate_finger_flexion(arr_lmlist[9], arr_lmlist[10], arr_lmlist[11])
            # print("Angolo medio: ", angle_middle)

            # Angolo per l'anulare
            angle_ring = calculate_finger_flexion(arr_lmlist[13], arr_lmlist[14], arr_lmlist[15])
            # print("Angolo anulare: ", angle_ring)

            # Angolo per il mignolo
            angle_pinky = calculate_finger_flexion(arr_lmlist[17], arr_lmlist[18], arr_lmlist[19])
            # print("Angolo mignolo: ", angle_pinky)
            
            #cv.circle(frame, (20, 30), 5, (0, 0, 255), -1)

            # print("this is pollice: ", arr_lmlist[2][2])
            # print("this is indice: ", arr_lmlist[5][2])
            # print("this is medio: ", arr_lmlist[9][2])
            # print("this is anulare: ", arr_lmlist[13][2])
            # print("this is mignolo: ", arr_lmlist[17][2])

            # Visualizza gli angoli sullo schermo
            # cv.putText(frame, f'Pollice: {int(angle_thumb)}°', (20, 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1, cv.LINE_AA)
            # cv.putText(frame, f'Indice: {int(angle_index)}°', (20, 60), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv.LINE_AA)
            # cv.putText(frame, f'Medio: {int(angle_middle)}°', (20, 90), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv.LINE_AA)
            # cv.putText(frame, f'Anulare: {int(angle_ring)}°', (20, 120), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv.LINE_AA)
            # cv.putText(frame, f'Mignolo: {int(angle_pinky)}°', (20, 150), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv.LINE_AA)

        # Mostra la conta delle dita
        # if fingerUp == [0, 0, 0, 0, 0]:
        #      cv.putText(frame, "Finger count: 0", (20, 460), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv.LINE_AA)
        # elif fingerUp == [0, 1, 0, 0, 0]:
        #      cv.putText(frame, "Finger count: 1", (20, 460), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv.LINE_AA)
        # elif fingerUp == [0, 1, 1, 0, 0]:
        #      cv.putText(frame, "Finger count: 2", (20, 460), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv.LINE_AA)
        # elif fingerUp == [0, 1, 1, 1, 0]:
        #      cv.putText(frame, "Finger count: 3", (20, 460), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv.LINE_AA)
        # elif fingerUp == [0, 1, 1, 1, 1]:
        #      cv.putText(frame, "Finger count: 4", (20, 460), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv.LINE_AA)
        # elif fingerUp == [1, 1, 1, 1, 1]:
        #      cv.putText(frame, "Finger count: 5", (20, 460), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv.LINE_AA)

    # Mostra il frame con gli angoli e la conta delle dita
    cv.imshow("Hand Tracking", frame)

    k = cv.waitKey(1)
    if k == ord("k"):
        break

video.release()
cv.destroyAllWindows()