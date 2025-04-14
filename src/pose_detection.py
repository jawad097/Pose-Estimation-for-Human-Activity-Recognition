import cv2
import mediapipe as mp
import numpy as np
import time
from pose_utils import (
    mp_pose, mp_drawing, mp_drawing_styles,
    extract_landmarks, extract_features, 
    draw_activity_text, draw_angles
)
from activity_classifier import ActivityClassifier

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Get webcam properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Webcam initialized: {width}x{height} at {fps} FPS")
    
    # Initialize MediaPipe Pose
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1  # 0, 1, or 2. Higher is more accurate but slower
    ) as pose:
        
        # Initialize activity classifier
        classifier = ActivityClassifier(history_size=30)
        
        # Initialize FPS calculation
        prev_time = time.time()
        frame_count = 0
        
        print("Starting pose detection. Press 'q' to quit.")
        
        while cap.isOpened():
            # Read frame from webcam
            success, image = cap.read()
            if not success:
                print("Error: Failed to read frame from webcam.")
                break
            
            # Flip the image horizontally for a selfie-view display
            image = cv2.flip(image, 1)
            
            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image and detect pose
            results = pose.process(image_rgb)
            
            # Draw pose landmarks on the image
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # Extract landmarks and features
                landmarks_dict = extract_landmarks(results)
                features = extract_features(landmarks_dict)
                
                # Classify activity
                activity = classifier.update(features)
                
                # Draw activity text
                image = draw_activity_text(image, activity, pos=(20, 50), font_scale=1, thickness=2)
                
                # Draw joint angles
                image = draw_angles(image, features, landmarks_dict, width, height)
                
                # Draw confidence scores
                confidence = classifier.get_activity_confidence()
                y_pos = 80
                for act, conf in confidence.items():
                    if conf > 0.05:  # Only show activities with confidence > 5%
                        cv2.putText(image, f"{act}: {conf:.2f}", (20, y_pos), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                        y_pos += 25
            
            # Calculate and display FPS
            frame_count += 1
            current_time = time.time()
            elapsed = current_time - prev_time
            
            if elapsed >= 1.0:  # Update FPS every second
                fps = frame_count / elapsed
                frame_count = 0
                prev_time = current_time
            
            cv2.putText(image, f"FPS: {fps:.1f}", (width - 120, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Display the image
            cv2.imshow('Pose Estimation for Activity Recognition', image)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Pose detection stopped.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        # Release resources in case of error
        cv2.destroyAllWindows()
