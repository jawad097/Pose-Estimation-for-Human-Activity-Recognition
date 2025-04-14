import numpy as np
from collections import deque

class ActivityClassifier:
    """
    A simple rule-based classifier for human activities based on pose data
    """
    
    def __init__(self, history_size=30):
        """
        Initialize the classifier
        
        Args:
            history_size: Number of frames to keep in history for temporal analysis
        """
        self.history = deque(maxlen=history_size)
        self.current_activity = "Unknown"
        self.activity_counter = {"Unknown": 0, "Standing": 0, "Sitting": 0, "Walking/Moving": 0, "Raising Arms": 0}
        self.stability_threshold = 5  # Number of consistent frames needed to change activity
    
    def update(self, features):
        """
        Update the classifier with new features
        
        Args:
            features: Dictionary of features extracted from pose landmarks
            
        Returns:
            activity: Classified activity
        """
        if not features:
            return "Unknown"
        
        # Add features to history
        self.history.append(features)
        
        # Classify current frame
        frame_activity = self._classify_frame(features)
        
        # Update counter for the current frame's activity
        self.activity_counter[frame_activity] += 1
        
        # Decrease counters for other activities
        for activity in self.activity_counter:
            if activity != frame_activity:
                self.activity_counter[activity] = max(0, self.activity_counter[activity] - 1)
        
        # Check if we should change the current activity
        max_activity = max(self.activity_counter, key=self.activity_counter.get)
        if self.activity_counter[max_activity] >= self.stability_threshold:
            self.current_activity = max_activity
        
        return self.current_activity
    
    def _classify_frame(self, features):
        """
        Classify a single frame based on features
        
        Args:
            features: Dictionary of features
            
        Returns:
            activity: Classified activity
        """
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
        
        # Check for walking/moving
        # For simplicity, we'll just check if knees are bent
        if (left_knee_angle < 160 or right_knee_angle < 160) and hip_ankle_ratio > 1.3:
            # If we have enough history, check for movement patterns
            if len(self.history) > 5:
                # Calculate movement of hips over time
                if 'left_hip' in features and all('left_hip' in h for h in list(self.history)[-5:]):
                    hip_positions = [list(h.values())[0][:2] for h in list(self.history)[-5:] if 'left_hip' in h]
                    if hip_positions:
                        hip_movement = np.std(hip_positions, axis=0)
                        if np.mean(hip_movement) > 0.01:  # Threshold for movement
                            return "Walking/Moving"
            
            # If we don't have enough history or movement is not detected
            # but knees are bent, still classify as walking/moving
            return "Walking/Moving"
        
        # Default to standing
        return "Standing"
    
    def get_activity_confidence(self):
        """
        Get confidence scores for each activity
        
        Returns:
            confidence: Dictionary of confidence scores
        """
        total = sum(self.activity_counter.values())
        if total == 0:
            return {activity: 0.0 for activity in self.activity_counter}
        
        return {activity: count / total for activity, count in self.activity_counter.items()}
