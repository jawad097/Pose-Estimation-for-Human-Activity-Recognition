import cv2
import mediapipe as mp
import numpy as np
import argparse
import os
import pickle
import time
from collections import deque
from pose_utils import (
    mp_pose, mp_drawing, mp_drawing_styles,
    extract_landmarks, extract_features
)

def load_model(model_dir):
    """
    Load trained model, scaler, and feature names
    
    Args:
        model_dir: Directory containing model files
        
    Returns:
        model: Trained model
        scaler: Feature scaler
        feature_names: List of feature names
    """
    try:
        # Load model
        with open(os.path.join(model_dir, 'activity_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        with open(os.path.join(model_dir, 'feature_scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        
        # Load feature names
        with open(os.path.join(model_dir, 'feature_names.pkl'), 'rb') as f:
            feature_names = pickle.load(f)
        
        print(f"Model loaded from {model_dir}")
        print(f"Model expects {len(feature_names)} features")
        
        return model, scaler, feature_names
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

def prepare_features_for_inference(features, feature_names):
    """
    Prepare features for model inference
    
    Args:
        features: Dictionary of features
        feature_names: List of feature names expected by the model
        
    Returns:
        X: Feature vector
    """
    import pandas as pd
    
    # Create feature vector with zeros
    feature_values = [features.get(name, 0) for name in feature_names]
    
    # Create DataFrame with proper feature names
    X = pd.DataFrame([feature_values], columns=feature_names)
    
    return X

def real_time_inference(model, scaler, feature_names, source=0, confidence_threshold=0.6):
    """
    Perform real-time activity recognition using webcam or video
    
    Args:
        model: Trained model
        scaler: Feature scaler
        feature_names: List of feature names
        source: Camera index or video file path
        confidence_threshold: Minimum confidence threshold for predictions
    """
    # Initialize video capture
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
        is_webcam = True
        print(f"Using webcam {source}")
    else:
        cap = cv2.VideoCapture(source)
        is_webcam = False
        print(f"Using video file: {source}")
    
    # Check if video source is opened correctly
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize MediaPipe Pose
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as pose:
        
        # Initialize variables
        start_time = time.time()
        frame_count = 0
        
        # Initialize prediction history for smoothing
        prediction_history = deque(maxlen=10)
        confidence_history = deque(maxlen=10)
        
        print("Starting real-time inference. Press 'q' to quit.")
        
        # Main loop
        while cap.isOpened():
            # Read frame
            success, image = cap.read()
            if not success:
                break
            
            # Flip the image horizontally for a selfie-view display if using webcam
            if is_webcam:
                image = cv2.flip(image, 1)
            
            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image and detect pose
            results = pose.process(image_rgb)
            
            # Initialize variables for this frame
            current_activity = "Unknown"
            confidence = 0.0
            
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
                
                # Prepare features for model inference
                X = prepare_features_for_inference(features, feature_names)
                
                # Scale features
                X_scaled = scaler.transform(X)
                
                # Make prediction
                prediction = model.predict(X_scaled)[0]
                
                # Get prediction probabilities
                probabilities = model.predict_proba(X_scaled)[0]
                confidence = np.max(probabilities)
                
                # Add to history for smoothing
                prediction_history.append(prediction)
                confidence_history.append(confidence)
                
                # Get most common prediction from history
                if len(prediction_history) > 0:
                    predictions, counts = np.unique(prediction_history, return_counts=True)
                    most_common_idx = np.argmax(counts)
                    current_activity = predictions[most_common_idx]
                    
                    # Calculate average confidence for the most common prediction
                    confidence = np.mean([
                        conf for pred, conf in zip(prediction_history, confidence_history)
                        if pred == current_activity
                    ])
            
            # Display activity and confidence
            if confidence >= confidence_threshold:
                activity_text = f"Activity: {current_activity} ({confidence:.2f})"
                cv2.putText(image, activity_text, (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, "Activity: Uncertain", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Calculate and display FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps_text = f"FPS: {frame_count / elapsed:.1f}"
                frame_count = 0
                start_time = time.time()
            else:
                fps_text = f"FPS: {fps:.1f}"
            
            cv2.putText(image, fps_text, (width - 150, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Display the image
            cv2.imshow('Activity Recognition', image)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Inference stopped.")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Real-time activity recognition using trained model')
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Directory containing model files')
    parser.add_argument('--source', '-s', type=str, default="0",
                        help='Camera index (e.g., 0) or video file path')
    parser.add_argument('--threshold', '-t', type=float, default=0.6,
                        help='Confidence threshold for predictions (default: 0.6)')
    args = parser.parse_args()
    
    # Load model
    model, scaler, feature_names = load_model(args.model)
    if model is None:
        return
    
    # Convert source to int if it's a number (camera index)
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    # Perform real-time inference
    real_time_inference(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        source=source,
        confidence_threshold=args.threshold
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInference interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        # Release resources in case of error
        cv2.destroyAllWindows()
