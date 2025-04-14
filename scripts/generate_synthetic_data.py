import numpy as np
import pandas as pd
import os
import argparse
import random
from datetime import datetime, timedelta

def generate_landmark_data(num_samples=1000, noise_level=0.05):
    """
    Generate synthetic landmark data for different activities
    
    Args:
        num_samples: Number of samples to generate per activity
        noise_level: Level of random noise to add to the data
        
    Returns:
        df: Pandas DataFrame with synthetic pose data
    """
    # Define activities and their characteristic features
    activities = {
        "standing": {
            "left_knee_angle": (170, 180),  # (min, max)
            "right_knee_angle": (170, 180),
            "left_elbow_angle": (150, 170),
            "right_elbow_angle": (150, 170),
            "left_wrist_to_shoulder_y": (-0.1, 0.1),
            "right_wrist_to_shoulder_y": (-0.1, 0.1),
            "hip_ankle_ratio": (1.5, 1.8)
        },
        "sitting": {
            "left_knee_angle": (80, 110),
            "right_knee_angle": (80, 110),
            "left_elbow_angle": (120, 160),
            "right_elbow_angle": (120, 160),
            "left_wrist_to_shoulder_y": (-0.1, 0.1),
            "right_wrist_to_shoulder_y": (-0.1, 0.1),
            "hip_ankle_ratio": (0.8, 1.2)
        },
        "walking": {
            "left_knee_angle": (140, 175),
            "right_knee_angle": (140, 175),
            "left_elbow_angle": (140, 160),
            "right_elbow_angle": (140, 160),
            "left_wrist_to_shoulder_y": (-0.1, 0.1),
            "right_wrist_to_shoulder_y": (-0.1, 0.1),
            "hip_ankle_ratio": (1.3, 1.6)
        },
        "raising_arms": {
            "left_knee_angle": (170, 180),
            "right_knee_angle": (170, 180),
            "left_elbow_angle": (150, 170),
            "right_elbow_angle": (150, 170),
            "left_wrist_to_shoulder_y": (0.2, 0.5),
            "right_wrist_to_shoulder_y": (0.2, 0.5),
            "hip_ankle_ratio": (1.5, 1.8)
        },
        "squatting": {
            "left_knee_angle": (80, 120),
            "right_knee_angle": (80, 120),
            "left_elbow_angle": (140, 160),
            "right_elbow_angle": (140, 160),
            "left_wrist_to_shoulder_y": (-0.1, 0.1),
            "right_wrist_to_shoulder_y": (-0.1, 0.1),
            "hip_ankle_ratio": (1.2, 1.5)
        },
        "lunging": {
            "left_knee_angle": (90, 130),
            "right_knee_angle": (150, 170),
            "left_elbow_angle": (140, 160),
            "right_elbow_angle": (140, 160),
            "left_wrist_to_shoulder_y": (-0.1, 0.1),
            "right_wrist_to_shoulder_y": (-0.1, 0.1),
            "hip_ankle_ratio": (1.3, 1.6)
        }
    }
    
    # Initialize empty list to store data
    data = []
    
    # Generate timestamp base
    base_timestamp = datetime.now()
    
    # Generate data for each activity
    for activity, features in activities.items():
        print(f"Generating {num_samples} samples for activity: {activity}")
        
        for i in range(num_samples):
            # Generate timestamp and frame_id
            timestamp = (base_timestamp + timedelta(seconds=i/30)).timestamp()
            frame_id = i
            
            # Initialize row with timestamp and frame_id
            row = {
                "timestamp": timestamp,
                "frame_id": frame_id
            }
            
            # Generate landmark data (simplified, just placeholders)
            for j in range(33):  # MediaPipe has 33 pose landmarks
                # Generate random values with some correlation to the activity
                x = random.uniform(0.2, 0.8) + random.uniform(-noise_level, noise_level)
                y = random.uniform(0.2, 0.8) + random.uniform(-noise_level, noise_level)
                z = random.uniform(-0.1, 0.1) + random.uniform(-noise_level, noise_level)
                visibility = random.uniform(0.8, 1.0) + random.uniform(-noise_level, noise_level)
                
                # Add to row
                row[f"landmark_{j}_x"] = x
                row[f"landmark_{j}_y"] = y
                row[f"landmark_{j}_z"] = z
                row[f"landmark_{j}_visibility"] = visibility
            
            # Generate feature data based on activity characteristics
            for feature, (min_val, max_val) in features.items():
                # Add some random noise to make it more realistic
                value = random.uniform(min_val, max_val) + random.uniform(-noise_level, noise_level)
                row[feature] = value
            
            # Add activity label
            row["activity"] = activity
            
            # Add row to data
            data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate synthetic pose data for training')
    parser.add_argument('--output', '-o', type=str, default="data/synthetic_data.csv",
                        help='Path to save the generated data (default: data/synthetic_data.csv)')
    parser.add_argument('--samples', '-s', type=int, default=500,
                        help='Number of samples to generate per activity (default: 500)')
    parser.add_argument('--noise', '-n', type=float, default=0.05,
                        help='Level of random noise to add to the data (default: 0.05)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Generate data
    df = generate_landmark_data(
        num_samples=args.samples,
        noise_level=args.noise
    )
    
    # Save data
    df.to_csv(args.output, index=False)
    
    print(f"Generated {len(df)} samples of synthetic data")
    print(f"Data saved to {args.output}")
    
    # Print activity distribution
    activity_counts = df['activity'].value_counts()
    print("\nActivity distribution:")
    for activity, count in activity_counts.items():
        print(f"  {activity}: {count} samples")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
