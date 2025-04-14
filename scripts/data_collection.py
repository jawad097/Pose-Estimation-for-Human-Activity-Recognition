import cv2
import mediapipe as mp
import numpy as np
import argparse
import os
import time
import csv
from pose_utils import (
    mp_pose, mp_drawing, mp_drawing_styles,
    extract_landmarks, extract_features
)

def collect_pose_data(source, output_file, activity_label=None, duration=60, sample_rate=1):
    """
    Collect pose data from webcam or video file and save to CSV
    
    Args:
        source: Camera index (int) or video file path (str)
        output_file: Path to save the collected data
        activity_label: Label for the activity being performed
        duration: Duration to collect data in seconds (for webcam)
        sample_rate: Process every Nth frame
    """
    # Initialize video capture
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
        is_webcam = True
        print(f"Collecting data from webcam {source}")
    else:
        cap = cv2.VideoCapture(source)
        is_webcam = False
        print(f"Collecting data from video file: {source}")
    
    # Check if video source is opened correctly
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    if not is_webcam:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video properties: {width}x{height} at {fps} FPS, {total_frames} frames")
    else:
        print(f"Webcam properties: {width}x{height} at {fps} FPS")
        print(f"Will collect data for {duration} seconds")
    
    # Initialize MediaPipe Pose
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as pose:
        
        # Prepare CSV file
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        with open(output_file, 'w', newline='') as csvfile:
            # Create CSV writer
            csv_writer = csv.writer(csvfile)
            
            # Write header row
            header = ['timestamp', 'frame_id']
            
            # Add landmark columns (x, y, z, visibility for each landmark)
            for i in range(33):  # MediaPipe has 33 pose landmarks
                header.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z', f'landmark_{i}_visibility'])
            
            # Add feature columns
            feature_names = ['left_elbow_angle', 'right_elbow_angle', 'left_knee_angle', 'right_knee_angle',
                            'left_wrist_to_shoulder_y', 'right_wrist_to_shoulder_y', 'hip_ankle_ratio']
            
            header.extend(feature_names)
            
            # Add activity label column
            header.append('activity')
            
            # Write header to CSV
            csv_writer.writerow(header)
            
            # Initialize variables
            start_time = time.time()
            frame_count = 0
            processed_count = 0
            
            print("Starting data collection. Press 'q' to stop.")
            
            # Main loop
            while cap.isOpened():
                # Read frame
                success, image = cap.read()
                if not success:
                    break
                
                # Check if we've reached the duration limit for webcam
                if is_webcam and (time.time() - start_time) > duration:
                    print(f"Reached duration limit of {duration} seconds")
                    break
                
                # Process every Nth frame
                if frame_count % sample_rate == 0:
                    # Flip the image horizontally for a selfie-view display if using webcam
                    if is_webcam:
                        image = cv2.flip(image, 1)
                    
                    # Convert the BGR image to RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Process the image and detect pose
                    results = pose.process(image_rgb)
                    
                    # Draw pose landmarks on the image for visualization
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
                        
                        # Prepare row data
                        row = [time.time(), frame_count]
                        
                        # Add all landmark data
                        for i in range(33):
                            if i in [landmark.value for landmark in mp_pose.PoseLandmark]:
                                landmark = results.pose_landmarks.landmark[i]
                                row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                            else:
                                row.extend([0, 0, 0, 0])  # Placeholder for missing landmarks
                        
                        # Add features
                        for feature in feature_names:
                            row.append(features.get(feature, 0))
                        
                        # Add activity label
                        row.append(activity_label if activity_label else "unknown")
                        
                        # Write row to CSV
                        csv_writer.writerow(row)
                        
                        processed_count += 1
                    
                    # Display status on the image
                    status_text = f"Recording: {processed_count} frames"
                    cv2.putText(image, status_text, (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    if activity_label:
                        cv2.putText(image, f"Activity: {activity_label}", (20, 90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    # Display the image
                    cv2.imshow('Pose Data Collection', image)
                
                # Update frame count
                frame_count += 1
                
                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Data collection completed. Processed {processed_count} frames.")
    print(f"Data saved to {output_file}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Collect pose data from webcam or video file')
    parser.add_argument('--source', '-s', type=str, default="0",
                        help='Camera index (e.g., 0) or video file path')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Path to save the collected data (CSV file)')
    parser.add_argument('--activity', '-a', type=str, default=None,
                        help='Label for the activity being performed')
    parser.add_argument('--duration', '-d', type=int, default=60,
                        help='Duration to collect data in seconds (for webcam)')
    parser.add_argument('--sample-rate', '-r', type=int, default=1,
                        help='Process every Nth frame')
    args = parser.parse_args()
    
    # Convert source to int if it's a number (camera index)
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    # Collect data
    collect_pose_data(
        source=source,
        output_file=args.output,
        activity_label=args.activity,
        duration=args.duration,
        sample_rate=args.sample_rate
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nData collection interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        # Release resources in case of error
        cv2.destroyAllWindows()
