import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hand_processor = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def calculate_finger_flexion(base, joint, tip, is_thumb=False):
    """
    Calculate finger flexion angle (0° = extended, 90° = fully flexed).
    """
    p1 = np.array(base, dtype=np.float32)
    p2 = np.array(joint, dtype=np.float32)
    p3 = np.array(tip, dtype=np.float32)
    
    vec_base_to_joint = p2 - p1  # Proximal segment
    vec_joint_to_tip = p3 - p2   # Distal segment
    
    if is_thumb:
        # Use full 3D vectors for thumb to capture sideways flexion
        v1 = vec_base_to_joint
        v2 = vec_joint_to_tip
    else:
        # YZ plane for other fingers (downward flexion)
        v1 = vec_base_to_joint[1:]
        v2 = vec_joint_to_tip[1:]
    
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    
    if mag1 == 0 or mag2 == 0:
        return 0
    
    dot_product = np.dot(v1, v2)
    cos_angle = np.clip(dot_product / (mag1 * mag2), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    # For thumb, maximize flexion toward palm
    if is_thumb:
        # Use 3D cross product to determine direction
        cross = np.cross(v1, v2)
        # Check if thumb is moving toward palm (negative x direction in selfie view)
        if cross[1] > 0:  # Adjust based on observed motion
            angle_deg = 180 - angle_deg  # Full angle if bending inward
        angle_deg = min(angle_deg, 90)  # Cap at 90°
    else:
        # Other fingers: downward bend is positive
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if cross < 0:
            angle_deg = -angle_deg
        angle_deg = max(0, angle_deg)
        angle_deg = min(angle_deg, 90)
    
    return angle_deg

def get_finger_angles(landmarks):
    finger_points = {
        "thumb": [2, 3, 4],
        "index": [5, 6, 8],
        "middle": [9, 10, 12],
        "ring": [13, 14, 16],
        "pinky": [17, 18, 20]
    }
    
    angles = {}
    for finger, indices in finger_points.items():
        base = [landmarks[indices[0]].x, landmarks[indices[0]].y, landmarks[indices[0]].z]
        joint = [landmarks[indices[1]].x, landmarks[indices[1]].y, landmarks[indices[1]].z]
        tip = [landmarks[indices[2]].x, landmarks[indices[2]].y, landmarks[indices[2]].z]
        
        is_thumb = (finger == "thumb")
        angle = calculate_finger_flexion(base, joint, tip, is_thumb)
        angles[finger] = angle
    
    return angles

def visualize_thumb(frame, landmarks):
    """Draw thumb vectors for debugging."""
    h, w = frame.shape[:2]
    base = (int(landmarks[2].x * w), int(landmarks[2].y * h))
    joint = (int(landmarks[3].x * w), int(landmarks[3].y * h))
    tip = (int(landmarks[4].x * w), int(landmarks[4].y * h))
    
    cv2.line(frame, base, joint, (255, 0, 0), 2)  # Blue: base to joint
    cv2.line(frame, joint, tip, (0, 255, 0), 2)   # Green: joint to tip

# Main loop
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    frame = cv2.flip(frame, 1)  # Flip for selfie view
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_processor.process(frame_rgb)
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[idx].classification[0].label
            
            if handedness == "Right":
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                visualize_thumb(frame, hand_landmarks.landmark)  # Debug thumb
                
                angles = get_finger_angles(hand_landmarks.landmark)
                
                y_pos = 30
                for finger, angle in angles.items():
                    text = f"{finger.capitalize()}: {angle:.1f}°"
                    cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                               0.7, (0, 255, 0), 2)
                    y_pos += 30
                
                cv2.putText(frame, "Right Hand", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 255, 255), 2)
    
    cv2.imshow("Right Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hand_processor.close()