import numpy as np
import math
import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define landmark indices for easier reference
LANDMARK_DICT = {
    'nose': 0,
    'left_eye_inner': 1,
    'left_eye': 2,
    'left_eye_outer': 3,
    'right_eye_inner': 4,
    'right_eye': 5,
    'right_eye_outer': 6,
    'left_ear': 7,
    'right_ear': 8,
    'mouth_left': 9,
    'mouth_right': 10,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_pinky': 17,
    'right_pinky': 18,
    'left_index': 19,
    'right_index': 20,
    'left_thumb': 21,
    'right_thumb': 22,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
    'left_heel': 29,
    'right_heel': 30,
    'left_foot_index': 31,
    'right_foot_index': 32
}

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points
    
    Args:
        a: First point [x, y]
        b: Mid point [x, y]
        c: End point [x, y]
        
    Returns:
        angle: Angle in degrees
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    # Calculate vectors
    ba = a - b
    bc = c - b
    
    # Calculate dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Ensure the value is in range [-1, 1]
    
    # Calculate angle in degrees
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)
    
    return angle

def extract_landmarks(results):
    """
    Extract landmarks from MediaPipe results
    
    Args:
        results: MediaPipe pose results
        
    Returns:
        landmarks_dict: Dictionary of landmark coordinates
    """
    landmarks_dict = {}
    
    if results.pose_landmarks:
        for name, index in LANDMARK_DICT.items():
            landmark = results.pose_landmarks.landmark[index]
            landmarks_dict[name] = [landmark.x, landmark.y, landmark.z, landmark.visibility]
    
    return landmarks_dict

def extract_features(landmarks_dict):
    """
    Extract features from landmarks for activity recognition
    
    Args:
        landmarks_dict: Dictionary of landmark coordinates
        
    Returns:
        features: Dictionary of features
    """
    features = {}
    
    if not landmarks_dict:
        return features
    
    # Calculate angles for key joints
    
    # Left elbow angle
    if all(k in landmarks_dict for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
        left_shoulder = landmarks_dict['left_shoulder'][:2]  # x, y
        left_elbow = landmarks_dict['left_elbow'][:2]
        left_wrist = landmarks_dict['left_wrist'][:2]
        features['left_elbow_angle'] = calculate_angle(left_shoulder, left_elbow, left_wrist)
    
    # Right elbow angle
    if all(k in landmarks_dict for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
        right_shoulder = landmarks_dict['right_shoulder'][:2]
        right_elbow = landmarks_dict['right_elbow'][:2]
        right_wrist = landmarks_dict['right_wrist'][:2]
        features['right_elbow_angle'] = calculate_angle(right_shoulder, right_elbow, right_wrist)
    
    # Left knee angle
    if all(k in landmarks_dict for k in ['left_hip', 'left_knee', 'left_ankle']):
        left_hip = landmarks_dict['left_hip'][:2]
        left_knee = landmarks_dict['left_knee'][:2]
        left_ankle = landmarks_dict['left_ankle'][:2]
        features['left_knee_angle'] = calculate_angle(left_hip, left_knee, left_ankle)
    
    # Right knee angle
    if all(k in landmarks_dict for k in ['right_hip', 'right_knee', 'right_ankle']):
        right_hip = landmarks_dict['right_hip'][:2]
        right_knee = landmarks_dict['right_knee'][:2]
        right_ankle = landmarks_dict['right_ankle'][:2]
        features['right_knee_angle'] = calculate_angle(right_hip, right_knee, right_ankle)
    
    # Calculate relative positions
    
    # Wrists relative to shoulders (for arm raising detection)
    if all(k in landmarks_dict for k in ['left_wrist', 'left_shoulder']):
        left_wrist_y = landmarks_dict['left_wrist'][1]
        left_shoulder_y = landmarks_dict['left_shoulder'][1]
        features['left_wrist_to_shoulder_y'] = left_shoulder_y - left_wrist_y
    
    if all(k in landmarks_dict for k in ['right_wrist', 'right_shoulder']):
        right_wrist_y = landmarks_dict['right_wrist'][1]
        right_shoulder_y = landmarks_dict['right_shoulder'][1]
        features['right_wrist_to_shoulder_y'] = right_shoulder_y - right_wrist_y
    
    # Hip to ankle height ratio (for sitting detection)
    if all(k in landmarks_dict for k in ['left_hip', 'left_ankle', 'left_shoulder']):
        left_hip_y = landmarks_dict['left_hip'][1]
        left_ankle_y = landmarks_dict['left_ankle'][1]
        left_shoulder_y = landmarks_dict['left_shoulder'][1]
        hip_to_ankle = abs(left_hip_y - left_ankle_y)
        shoulder_to_hip = abs(left_shoulder_y - left_hip_y)
        if shoulder_to_hip > 0:
            features['hip_ankle_ratio'] = hip_to_ankle / shoulder_to_hip
    
    return features

def classify_activity(features, history=None, window_size=10):
    """
    Classify activity based on extracted features
    
    Args:
        features: Dictionary of features
        history: List of previous feature dictionaries
        window_size: Number of frames to consider for temporal analysis
        
    Returns:
        activity: Classified activity
    """
    if not features:
        return "Unknown"
    
    # Simple rule-based classification
    
    # Check for arm raising
    left_arm_raised = features.get('left_wrist_to_shoulder_y', 0) > 0.1
    right_arm_raised = features.get('right_wrist_to_shoulder_y', 0) > 0.1
    
    if left_arm_raised and right_arm_raised:
        return "Raising Arms"
    
    # Check for sitting
    hip_ankle_ratio = features.get('hip_ankle_ratio', 1.5)
    left_knee_angle = features.get('left_knee_angle', 180)
    right_knee_angle = features.get('right_knee_angle', 180)
    
    if hip_ankle_ratio < 1.2 or (left_knee_angle < 110 and right_knee_angle < 110):
        return "Sitting"
    
    # Check for walking (would need temporal information)
    # For simplicity, we'll just check if knees are bent
    if (left_knee_angle < 160 or right_knee_angle < 160) and hip_ankle_ratio > 1.3:
        return "Walking/Moving"
    
    # Default to standing
    return "Standing"

def draw_activity_text(image, activity, pos=(50, 50), font_scale=1, thickness=2):
    """
    Draw activity text on the image
    
    Args:
        image: OpenCV image
        activity: Activity text
        pos: Position tuple (x, y)
        font_scale: Font scale
        thickness: Line thickness
        
    Returns:
        image: Image with text
    """
    cv2.putText(image, f"Activity: {activity}", pos, cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    return image

def draw_angles(image, features, landmarks_dict, width, height):
    """
    Draw joint angles on the image
    
    Args:
        image: OpenCV image
        features: Dictionary of features
        landmarks_dict: Dictionary of landmark coordinates
        width: Image width
        height: Image height
        
    Returns:
        image: Image with angles
    """
    # Draw elbow angles
    if 'left_elbow_angle' in features and 'left_elbow' in landmarks_dict:
        x, y = landmarks_dict['left_elbow'][:2]
        pos = (int(x * width), int(y * height))
        cv2.putText(image, f"{int(features['left_elbow_angle'])}°", pos, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    
    if 'right_elbow_angle' in features and 'right_elbow' in landmarks_dict:
        x, y = landmarks_dict['right_elbow'][:2]
        pos = (int(x * width), int(y * height))
        cv2.putText(image, f"{int(features['right_elbow_angle'])}°", pos, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    
    # Draw knee angles
    if 'left_knee_angle' in features and 'left_knee' in landmarks_dict:
        x, y = landmarks_dict['left_knee'][:2]
        pos = (int(x * width), int(y * height))
        cv2.putText(image, f"{int(features['left_knee_angle'])}°", pos, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    
    if 'right_knee_angle' in features and 'right_knee' in landmarks_dict:
        x, y = landmarks_dict['right_knee'][:2]
        pos = (int(x * width), int(y * height))
        cv2.putText(image, f"{int(features['right_knee_angle'])}°", pos, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    
    return image
