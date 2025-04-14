import cv2
import mediapipe as mp
import numpy as np
import argparse
import time
import os
import datetime
from collections import deque
from pose_utils import (
    mp_pose, mp_drawing, mp_drawing_styles,
    extract_landmarks, extract_features, calculate_angle
)

class ExerciseCounter:
    """
    Class to count exercise repetitions based on pose detection
    """
    
    def __init__(self, exercise_type, smoothing_window=10, confidence_threshold=0.6):
        """
        Initialize exercise counter
        
        Args:
            exercise_type: Type of exercise to count ('squat', 'pushup', 'arm_raise', 'lunge', 'plank')
            smoothing_window: Window size for smoothing angle values
            confidence_threshold: Minimum confidence for landmark detection
        """
        self.exercise_type = exercise_type.lower()
        self.rep_count = 0
        self.state = "up"  # or "down" depending on exercise
        self.smoothing_window = smoothing_window
        self.confidence_threshold = confidence_threshold
        
        # Initialize angle history for smoothing
        self.left_angle_history = deque(maxlen=smoothing_window)
        self.right_angle_history = deque(maxlen=smoothing_window)
        
        # Initialize state tracking
        self.state_history = deque(maxlen=5)  # Track last 5 states for stability
        self.rep_quality = []  # Track quality of each rep
        self.last_rep_time = time.time()
        self.rep_durations = []  # Track duration of each rep
        
        # Initialize calibration
        self.calibrated = False
        self.max_angle = 0
        self.min_angle = 180
        self.calibration_frames = 0
        self.calibration_max_frames = 30  # Frames to use for calibration
        
        # Set exercise-specific parameters
        self._set_exercise_parameters()
    
    def _set_exercise_parameters(self):
        """Set parameters specific to each exercise type"""
        # Default parameters
        self.up_angle_threshold = 150
        self.down_angle_threshold = 100
        self.joint_name = "joint"
        self.secondary_joint_name = None
        self.secondary_up_threshold = None
        self.secondary_down_threshold = None
        self.min_rep_duration = 1.0  # Minimum seconds for a quality rep
        self.max_rep_duration = 5.0  # Maximum seconds for a quality rep
        
        # Exercise-specific parameters
        if self.exercise_type == 'squat':
            self.up_angle_threshold = 150  # Knees almost straight
            self.down_angle_threshold = 100  # Knees bent
            self.joint_name = "knee"
            self.form_cues = [
                "Keep your back straight",
                "Knees should not go past toes",
                "Keep weight in heels"
            ]
        elif self.exercise_type == 'pushup':
            self.up_angle_threshold = 160  # Arms almost straight
            self.down_angle_threshold = 90  # Arms bent
            self.joint_name = "elbow"
            self.secondary_joint_name = "back"
            self.secondary_up_threshold = 170  # Back should be straight
            self.secondary_down_threshold = 160  # Back should remain straight
            self.form_cues = [
                "Keep your back straight",
                "Lower chest to ground",
                "Elbows at 45° angle"
            ]
        elif self.exercise_type == 'arm_raise':
            self.up_angle_threshold = 150  # Arms raised
            self.down_angle_threshold = 80  # Arms lowered
            self.joint_name = "shoulder"
            self.form_cues = [
                "Keep movements controlled",
                "Fully extend arms",
                "Keep core engaged"
            ]
        elif self.exercise_type == 'lunge':
            self.up_angle_threshold = 160  # Legs straight
            self.down_angle_threshold = 110  # Front knee bent
            self.joint_name = "knee"
            self.secondary_joint_name = "hip"
            self.secondary_up_threshold = 170  # Hip extended
            self.secondary_down_threshold = 140  # Hip flexed
            self.form_cues = [
                "Keep torso upright",
                "Front knee aligned with ankle",
                "Step far enough forward"
            ]
        elif self.exercise_type == 'plank':
            # Plank is isometric, so we track duration instead of reps
            self.joint_name = "elbow"
            self.secondary_joint_name = "back"
            self.up_angle_threshold = 90  # Elbows at 90 degrees
            self.down_angle_threshold = 80  # Minimum acceptable elbow angle
            self.secondary_up_threshold = 170  # Back should be straight
            self.secondary_down_threshold = 160  # Minimum acceptable back angle
            self.form_cues = [
                "Keep your back straight",
                "Engage your core",
                "Don't let hips sag"
            ]
        elif self.exercise_type == 'jumping_jack':
            self.up_angle_threshold = 160  # Arms and legs spread
            self.down_angle_threshold = 30  # Arms and legs together
            self.joint_name = "shoulder"
            self.secondary_joint_name = "hip"
            self.secondary_up_threshold = 140  # Legs spread
            self.secondary_down_threshold = 20  # Legs together
            self.form_cues = [
                "Jump with both feet",
                "Extend arms fully",
                "Keep movements synchronized"
            ]
        elif self.exercise_type == 'bicycle_crunch':
            self.up_angle_threshold = 90  # Knee to elbow
            self.down_angle_threshold = 160  # Leg extended
            self.joint_name = "knee"
            self.secondary_joint_name = "elbow"
            self.secondary_up_threshold = 90  # Elbow to knee
            self.secondary_down_threshold = 160  # Arm extended
            self.form_cues = [
                "Keep lower back pressed to floor",
                "Touch elbow to opposite knee",
                "Extend leg fully on each rep"
            ]
        elif self.exercise_type == 'mountain_climber':
            self.up_angle_threshold = 160  # Leg extended
            self.down_angle_threshold = 90  # Knee to chest
            self.joint_name = "knee"
            self.secondary_joint_name = "back"
            self.secondary_up_threshold = 170  # Back straight
            self.secondary_down_threshold = 160  # Back remains straight
            self.form_cues = [
                "Keep your back straight",
                "Bring knee to chest",
                "Maintain plank position"
            ]
        elif self.exercise_type == 'side_plank':
            self.joint_name = "shoulder"
            self.secondary_joint_name = "hip"
            self.up_angle_threshold = 170  # Body straight
            self.down_angle_threshold = 150  # Minimum acceptable angle
            self.secondary_up_threshold = 170  # Hip straight
            self.secondary_down_threshold = 150  # Hip straight
            self.form_cues = [
                "Keep body in straight line",
                "Stack feet or stagger them",
                "Raise free arm straight up"
            ]
        elif self.exercise_type == 'burpee':
            self.up_angle_threshold = 160  # Standing position
            self.down_angle_threshold = 90  # Squat position
            self.joint_name = "knee"
            self.secondary_joint_name = "elbow"
            self.secondary_up_threshold = 160  # Arms straight (pushup position)
            self.secondary_down_threshold = 90  # Arms bent (pushup position)
            self.form_cues = [
                "Full extension at top",
                "Chest to floor in pushup",
                "Jump at the top of each rep"
            ]
        elif self.exercise_type == 'tricep_dip':
            self.up_angle_threshold = 160  # Arms straight
            self.down_angle_threshold = 90  # Arms bent
            self.joint_name = "elbow"
            self.secondary_joint_name = "shoulder"
            self.secondary_up_threshold = 180  # Shoulders down
            self.secondary_down_threshold = 160  # Shoulders remain stable
            self.form_cues = [
                "Keep elbows pointing backward",
                "Lower until arms are 90°",
                "Keep shoulders down and back"
            ]
        elif self.exercise_type == 'calf_raise':
            self.up_angle_threshold = 120  # On toes
            self.down_angle_threshold = 90  # Flat feet
            self.joint_name = "ankle"
            self.secondary_joint_name = "knee"
            self.secondary_up_threshold = 170  # Knees straight
            self.secondary_down_threshold = 160  # Knees remain straight
            self.form_cues = [
                "Rise fully onto toes",
                "Keep knees straight",
                "Control the movement"
            ]
        else:
            raise ValueError(f"Unsupported exercise type: {self.exercise_type}")
    
    def update(self, landmarks_dict, landmark_confidence=None):
        """
        Update counter based on new landmarks
        
        Args:
            landmarks_dict: Dictionary of landmark coordinates
            landmark_confidence: Dictionary of landmark confidence values
            
        Returns:
            result: Dictionary with rep count, state, angles, and form feedback
        """
        # Initialize result dictionary
        result = {
            'rep_count': self.rep_count,
            'state': self.state,
            'primary_angle': None,
            'secondary_angle': None,
            'form_feedback': None,
            'rep_quality': None,
            'calibrated': self.calibrated
        }
        
        # Check if landmarks have sufficient confidence
        if landmark_confidence is None:
            landmark_confidence = {k: 1.0 for k in landmarks_dict.keys()}
        
        # Extract relevant angles based on exercise type
        primary_angle = self._get_primary_angle(landmarks_dict, landmark_confidence)
        secondary_angle = self._get_secondary_angle(landmarks_dict, landmark_confidence)
        
        # Update result with angles
        result['primary_angle'] = primary_angle
        result['secondary_angle'] = secondary_angle
        
        # If no valid angles detected, return early
        if primary_angle is None:
            return result
        
        # Calibration phase - determine user's range of motion
        if not self.calibrated and self.calibration_frames < self.calibration_max_frames:
            self.calibration_frames += 1
            self.max_angle = max(self.max_angle, primary_angle)
            self.min_angle = min(self.min_angle, primary_angle)
            
            # After collecting enough frames, set adaptive thresholds
            if self.calibration_frames >= self.calibration_max_frames:
                self._set_adaptive_thresholds()
                self.calibrated = True
                result['calibrated'] = True
        
        # Process exercise state and count reps
        if self.calibrated:
            # Get previous state
            prev_state = self.state
            
            # Update state based on angle
            if self.state == "up" and primary_angle < self.down_angle_threshold:
                self.state = "down"
                # Record time when entering down state for rep duration tracking
                if prev_state == "up":
                    self.last_rep_time = time.time()
            elif self.state == "down" and primary_angle > self.up_angle_threshold:
                self.state = "up"
                # Count rep and evaluate quality when returning to up state
                if prev_state == "down":
                    self.rep_count += 1
                    rep_duration = time.time() - self.last_rep_time
                    self.rep_durations.append(rep_duration)
                    
                    # Evaluate rep quality
                    quality = self._evaluate_rep_quality(primary_angle, secondary_angle, rep_duration)
                    self.rep_quality.append(quality)
                    result['rep_quality'] = quality
            
            # Add current state to history for stability
            self.state_history.append(self.state)
            
            # Ensure state is stable (majority of recent states)
            if len(self.state_history) >= 3:
                up_count = self.state_history.count("up")
                down_count = self.state_history.count("down")
                if up_count > down_count:
                    stable_state = "up"
                else:
                    stable_state = "down"
                
                # Only update if stable state is different from current
                if stable_state != self.state:
                    self.state = stable_state
            
            # Generate form feedback
            if secondary_angle is not None:
                result['form_feedback'] = self._generate_form_feedback(primary_angle, secondary_angle)
        
        # Update result with current state and rep count
        result['state'] = self.state
        result['rep_count'] = self.rep_count
        
        return result
    
    def _set_adaptive_thresholds(self):
        """Set adaptive thresholds based on user's range of motion"""
        # Ensure we have a reasonable range
        angle_range = self.max_angle - self.min_angle
        if angle_range > 20:  # Only adapt if range is significant
            # Set thresholds based on user's range of motion
            self.up_angle_threshold = self.max_angle - (angle_range * 0.2)  # 20% down from max
            self.down_angle_threshold = self.min_angle + (angle_range * 0.2)  # 20% up from min
            
            print(f"Calibration complete. Range: {self.min_angle:.1f}° - {self.max_angle:.1f}°")
            print(f"Adaptive thresholds: Down: {self.down_angle_threshold:.1f}°, Up: {self.up_angle_threshold:.1f}°")
    
    def _evaluate_rep_quality(self, primary_angle, secondary_angle, duration):
        """
        Evaluate the quality of a repetition
        
        Args:
            primary_angle: Primary joint angle
            secondary_angle: Secondary joint angle (if applicable)
            duration: Duration of the repetition in seconds
            
        Returns:
            quality: Quality score (0-100)
        """
        quality = 100  # Start with perfect score
        
        # Check rep duration
        if duration < self.min_rep_duration:
            # Too fast
            quality -= 30
        elif duration > self.max_rep_duration:
            # Too slow
            quality -= 20
        
        # Check range of motion
        angle_range = self.max_angle - self.min_angle
        expected_range = self.up_angle_threshold - self.down_angle_threshold
        if angle_range < expected_range * 0.8:
            # Insufficient range of motion
            quality -= 30
        
        # Check secondary angle if available
        if secondary_angle is not None and self.secondary_up_threshold is not None:
            if secondary_angle < self.secondary_down_threshold:
                # Poor form in secondary joint
                quality -= 20
        
        # Ensure quality is between 0 and 100
        quality = max(0, min(100, quality))
        
        return quality
    
    def _generate_form_feedback(self, primary_angle, secondary_angle):
        """
        Generate form feedback based on angles
        
        Args:
            primary_angle: Primary joint angle
            secondary_angle: Secondary joint angle (if applicable)
            
        Returns:
            feedback: Form feedback message
        """
        if self.exercise_type == 'squat':
            if primary_angle > 130 and self.state == "down":
                return "Squat deeper"
            elif primary_angle < 90 and self.state == "down":
                return "Don't squat too deep"
        elif self.exercise_type == 'pushup':
            if secondary_angle is not None and secondary_angle < 160:
                return "Keep your back straight"
            elif primary_angle > 100 and self.state == "down":
                return "Lower your chest more"
        elif self.exercise_type == 'plank':
            if secondary_angle is not None and secondary_angle < 160:
                return "Don't let your hips sag"
            elif primary_angle < 80 or primary_angle > 100:
                return "Keep elbows at 90 degrees"
        
        # If no specific feedback, return a random form cue
        if hasattr(self, 'form_cues') and self.form_cues:
            return np.random.choice(self.form_cues)
        
        return None
    
    def _get_primary_angle(self, landmarks_dict, landmark_confidence):
        """
        Get the primary angle for the exercise
        
        Args:
            landmarks_dict: Dictionary of landmark coordinates
            landmark_confidence: Dictionary of landmark confidence values
            
        Returns:
            angle: Primary joint angle for the exercise
        """
        # Try to get angles from both sides and use the more reliable one
        left_angle = None
        right_angle = None
        
        if self.exercise_type == 'squat':
            # Left knee angle (hip-knee-ankle)
            if all(k in landmarks_dict for k in ['left_hip', 'left_knee', 'left_ankle']):
                if all(landmark_confidence.get(k, 0) >= self.confidence_threshold for k in ['left_hip', 'left_knee', 'left_ankle']):
                    left_hip = landmarks_dict['left_hip'][:2]
                    left_knee = landmarks_dict['left_knee'][:2]
                    left_ankle = landmarks_dict['left_ankle'][:2]
                    left_angle = calculate_angle(left_hip, left_knee, left_ankle)
                    self.left_angle_history.append(left_angle)
            
            # Right knee angle (hip-knee-ankle)
            if all(k in landmarks_dict for k in ['right_hip', 'right_knee', 'right_ankle']):
                if all(landmark_confidence.get(k, 0) >= self.confidence_threshold for k in ['right_hip', 'right_knee', 'right_ankle']):
                    right_hip = landmarks_dict['right_hip'][:2]
                    right_knee = landmarks_dict['right_knee'][:2]
                    right_ankle = landmarks_dict['right_ankle'][:2]
                    right_angle = calculate_angle(right_hip, right_knee, right_ankle)
                    self.right_angle_history.append(right_angle)
        
        elif self.exercise_type == 'pushup':
            # Left elbow angle (shoulder-elbow-wrist)
            if all(k in landmarks_dict for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
                if all(landmark_confidence.get(k, 0) >= self.confidence_threshold for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
                    left_shoulder = landmarks_dict['left_shoulder'][:2]
                    left_elbow = landmarks_dict['left_elbow'][:2]
                    left_wrist = landmarks_dict['left_wrist'][:2]
                    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    self.left_angle_history.append(left_angle)
            
            # Right elbow angle (shoulder-elbow-wrist)
            if all(k in landmarks_dict for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
                if all(landmark_confidence.get(k, 0) >= self.confidence_threshold for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
                    right_shoulder = landmarks_dict['right_shoulder'][:2]
                    right_elbow = landmarks_dict['right_elbow'][:2]
                    right_wrist = landmarks_dict['right_wrist'][:2]
                    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                    self.right_angle_history.append(right_angle)
        
        elif self.exercise_type == 'arm_raise':
            # Left shoulder angle (hip-shoulder-elbow)
            if all(k in landmarks_dict for k in ['left_hip', 'left_shoulder', 'left_elbow']):
                if all(landmark_confidence.get(k, 0) >= self.confidence_threshold for k in ['left_hip', 'left_shoulder', 'left_elbow']):
                    left_hip = landmarks_dict['left_hip'][:2]
                    left_shoulder = landmarks_dict['left_shoulder'][:2]
                    left_elbow = landmarks_dict['left_elbow'][:2]
                    left_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
                    self.left_angle_history.append(left_angle)
            
            # Right shoulder angle (hip-shoulder-elbow)
            if all(k in landmarks_dict for k in ['right_hip', 'right_shoulder', 'right_elbow']):
                if all(landmark_confidence.get(k, 0) >= self.confidence_threshold for k in ['right_hip', 'right_shoulder', 'right_elbow']):
                    right_hip = landmarks_dict['right_hip'][:2]
                    right_shoulder = landmarks_dict['right_shoulder'][:2]
                    right_elbow = landmarks_dict['right_elbow'][:2]
                    right_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
                    self.right_angle_history.append(right_angle)
        
        elif self.exercise_type == 'lunge':
            # Left knee angle (hip-knee-ankle)
            if all(k in landmarks_dict for k in ['left_hip', 'left_knee', 'left_ankle']):
                if all(landmark_confidence.get(k, 0) >= self.confidence_threshold for k in ['left_hip', 'left_knee', 'left_ankle']):
                    left_hip = landmarks_dict['left_hip'][:2]
                    left_knee = landmarks_dict['left_knee'][:2]
                    left_ankle = landmarks_dict['left_ankle'][:2]
                    left_angle = calculate_angle(left_hip, left_knee, left_ankle)
                    self.left_angle_history.append(left_angle)
            
            # Right knee angle (hip-knee-ankle)
            if all(k in landmarks_dict for k in ['right_hip', 'right_knee', 'right_ankle']):
                if all(landmark_confidence.get(k, 0) >= self.confidence_threshold for k in ['right_hip', 'right_knee', 'right_ankle']):
                    right_hip = landmarks_dict['right_hip'][:2]
                    right_knee = landmarks_dict['right_knee'][:2]
                    right_ankle = landmarks_dict['right_ankle'][:2]
                    right_angle = calculate_angle(right_hip, right_knee, right_ankle)
                    self.right_angle_history.append(right_angle)
        
        elif self.exercise_type == 'plank':
            # Left elbow angle (shoulder-elbow-wrist)
            if all(k in landmarks_dict for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
                if all(landmark_confidence.get(k, 0) >= self.confidence_threshold for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
                    left_shoulder = landmarks_dict['left_shoulder'][:2]
                    left_elbow = landmarks_dict['left_elbow'][:2]
                    left_wrist = landmarks_dict['left_wrist'][:2]
                    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    self.left_angle_history.append(left_angle)
            
            # Right elbow angle (shoulder-elbow-wrist)
            if all(k in landmarks_dict for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
                if all(landmark_confidence.get(k, 0) >= self.confidence_threshold for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
                    right_shoulder = landmarks_dict['right_shoulder'][:2]
                    right_elbow = landmarks_dict['right_elbow'][:2]
                    right_wrist = landmarks_dict['right_wrist'][:2]
                    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                    self.right_angle_history.append(right_angle)
        
        # Calculate smoothed angles if available
        left_smoothed = np.mean(self.left_angle_history) if len(self.left_angle_history) > 0 else None
        right_smoothed = np.mean(self.right_angle_history) if len(self.right_angle_history) > 0 else None
        
        # Choose the best angle (prioritize the side with more history data)
        if left_smoothed is not None and right_smoothed is not None:
            # Use average of both sides if both are available
            return (left_smoothed + right_smoothed) / 2
        elif left_smoothed is not None:
            return left_smoothed
        elif right_smoothed is not None:
            return right_smoothed
        
        # If no smoothed angles available, use the raw angles
        if left_angle is not None and right_angle is not None:
            return (left_angle + right_angle) / 2
        elif left_angle is not None:
            return left_angle
        elif right_angle is not None:
            return right_angle
        
        return None
    
    def _get_secondary_angle(self, landmarks_dict, landmark_confidence):
        """
        Get the secondary angle for form checking
        
        Args:
            landmarks_dict: Dictionary of landmark coordinates
            landmark_confidence: Dictionary of landmark confidence values
            
        Returns:
            angle: Secondary joint angle for form checking
        """
        if self.secondary_joint_name is None:
            return None
        
        if self.secondary_joint_name == 'back':
            # Calculate back angle (shoulder-hip-knee)
            if all(k in landmarks_dict for k in ['right_shoulder', 'right_hip', 'right_knee']):
                if all(landmark_confidence.get(k, 0) >= self.confidence_threshold for k in ['right_shoulder', 'right_hip', 'right_knee']):
                    right_shoulder = landmarks_dict['right_shoulder'][:2]
                    right_hip = landmarks_dict['right_hip'][:2]
                    right_knee = landmarks_dict['right_knee'][:2]
                    return calculate_angle(right_shoulder, right_hip, right_knee)
            
            if all(k in landmarks_dict for k in ['left_shoulder', 'left_hip', 'left_knee']):
                if all(landmark_confidence.get(k, 0) >= self.confidence_threshold for k in ['left_shoulder', 'left_hip', 'left_knee']):
                    left_shoulder = landmarks_dict['left_shoulder'][:2]
                    left_hip = landmarks_dict['left_hip'][:2]
                    left_knee = landmarks_dict['left_knee'][:2]
                    return calculate_angle(left_shoulder, left_hip, left_knee)
        
        elif self.secondary_joint_name == 'hip':
            # Calculate hip angle (shoulder-hip-knee)
            if all(k in landmarks_dict for k in ['right_shoulder', 'right_hip', 'right_knee']):
                if all(landmark_confidence.get(k, 0) >= self.confidence_threshold for k in ['right_shoulder', 'right_hip', 'right_knee']):
                    right_shoulder = landmarks_dict['right_shoulder'][:2]
                    right_hip = landmarks_dict['right_hip'][:2]
                    right_knee = landmarks_dict['right_knee'][:2]
                    return calculate_angle(right_shoulder, right_hip, right_knee)
            
            if all(k in landmarks_dict for k in ['left_shoulder', 'left_hip', 'left_knee']):
                if all(landmark_confidence.get(k, 0) >= self.confidence_threshold for k in ['left_shoulder', 'left_hip', 'left_knee']):
                    left_shoulder = landmarks_dict['left_shoulder'][:2]
                    left_hip = landmarks_dict['left_hip'][:2]
                    left_knee = landmarks_dict['left_knee'][:2]
                    return calculate_angle(left_shoulder, left_hip, left_knee)
        
        return None
    
    def get_statistics(self):
        """
        Get exercise statistics
        
        Returns:
            stats: Dictionary with exercise statistics
        """
        stats = {
            'rep_count': self.rep_count,
            'avg_quality': np.mean(self.rep_quality) if self.rep_quality else 0,
            'avg_duration': np.mean(self.rep_durations) if self.rep_durations else 0,
            'range_of_motion': (self.max_angle - self.min_angle) if self.calibrated else 0
        }
        
        return stats
    
    def reset(self):
        """Reset the counter"""
        self.rep_count = 0
        self.state = "up"
        self.left_angle_history.clear()
        self.right_angle_history.clear()
        self.state_history.clear()
        self.rep_quality = []
        self.rep_durations = []
        self.calibrated = False
        self.calibration_frames = 0

def show_exercise_demo(exercise_type, width, height):
    """
    Create a demonstration image for the exercise
    
    Args:
        exercise_type: Type of exercise to demonstrate
        width: Width of the image
        height: Height of the image
        
    Returns:
        demo_image: Image with exercise demonstration
    """
    # Create a blank image with dark background
    demo_image = np.zeros((height, width, 3), dtype=np.uint8)
    demo_image[:] = (50, 50, 50)  # Dark gray background
    
    # Add title
    title = f"{exercise_type.upper()} DEMONSTRATION"
    cv2.putText(demo_image, title, (int(width/2) - 200, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Add exercise-specific instructions and visualization
    if exercise_type == 'squat':
        instructions = [
            "1. Stand with feet shoulder-width apart",
            "2. Keep your back straight",
            "3. Bend your knees and lower your body",
            "4. Keep weight in your heels",
            "5. Return to standing position",
            "6. Repeat the movement"
        ]
        # Draw simple stick figure for squat
        # Standing position
        cv2.circle(demo_image, (int(width/4), 150), 20, (0, 255, 0), -1)  # Head
        cv2.line(demo_image, (int(width/4), 170), (int(width/4), 250), (0, 255, 0), 3)  # Body
        cv2.line(demo_image, (int(width/4), 250), (int(width/4) - 40, 350), (0, 255, 0), 3)  # Left leg
        cv2.line(demo_image, (int(width/4), 250), (int(width/4) + 40, 350), (0, 255, 0), 3)  # Right leg
        cv2.line(demo_image, (int(width/4), 200), (int(width/4) - 50, 220), (0, 255, 0), 3)  # Left arm
        cv2.line(demo_image, (int(width/4), 200), (int(width/4) + 50, 220), (0, 255, 0), 3)  # Right arm
        
        # Squat position
        cv2.circle(demo_image, (int(width*3/4), 200), 20, (0, 255, 0), -1)  # Head
        cv2.line(demo_image, (int(width*3/4), 220), (int(width*3/4), 270), (0, 255, 0), 3)  # Body
        cv2.line(demo_image, (int(width*3/4), 270), (int(width*3/4) - 60, 320), (0, 255, 0), 3)  # Left leg
        cv2.line(demo_image, (int(width*3/4), 270), (int(width*3/4) + 60, 320), (0, 255, 0), 3)  # Right leg
        cv2.line(demo_image, (int(width*3/4), 240), (int(width*3/4) - 50, 260), (0, 255, 0), 3)  # Left arm
        cv2.line(demo_image, (int(width*3/4), 240), (int(width*3/4) + 50, 260), (0, 255, 0), 3)  # Right arm
        
        # Add labels
        cv2.putText(demo_image, "Starting Position", (int(width/4) - 80, 400), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(demo_image, "Squat Position", (int(width*3/4) - 80, 400), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
    elif exercise_type == 'pushup':
        instructions = [
            "1. Start in plank position with arms straight",
            "2. Keep your body in a straight line",
            "3. Lower your body by bending your elbows",
            "4. Keep elbows at about 45° angle",
            "5. Push back up to starting position",
            "6. Repeat the movement"
        ]
        # Draw simple stick figure for pushup
        # Up position
        cv2.circle(demo_image, (int(width/4), 150), 15, (0, 255, 0), -1)  # Head
        cv2.line(demo_image, (int(width/4), 165), (int(width/4), 250), (0, 255, 0), 3)  # Body
        cv2.line(demo_image, (int(width/4) - 60, 180), (int(width/4), 180), (0, 255, 0), 3)  # Left arm
        cv2.line(demo_image, (int(width/4), 180), (int(width/4) + 60, 180), (0, 255, 0), 3)  # Right arm
        cv2.line(demo_image, (int(width/4), 250), (int(width/4) - 100, 250), (0, 255, 0), 3)  # Left leg
        cv2.line(demo_image, (int(width/4), 250), (int(width/4) + 100, 250), (0, 255, 0), 3)  # Right leg
        
        # Down position
        cv2.circle(demo_image, (int(width*3/4), 200), 15, (0, 255, 0), -1)  # Head
        cv2.line(demo_image, (int(width*3/4), 215), (int(width*3/4), 300), (0, 255, 0), 3)  # Body
        cv2.line(demo_image, (int(width*3/4) - 40, 230), (int(width*3/4), 200), (0, 255, 0), 3)  # Left arm bent
        cv2.line(demo_image, (int(width*3/4), 200), (int(width*3/4) + 40, 230), (0, 255, 0), 3)  # Right arm bent
        cv2.line(demo_image, (int(width*3/4), 300), (int(width*3/4) - 100, 300), (0, 255, 0), 3)  # Left leg
        cv2.line(demo_image, (int(width*3/4), 300), (int(width*3/4) + 100, 300), (0, 255, 0), 3)  # Right leg
        
        # Add labels
        cv2.putText(demo_image, "Up Position", (int(width/4) - 60, 300), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(demo_image, "Down Position", (int(width*3/4) - 70, 350), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
    elif exercise_type == 'arm_raise':
        instructions = [
            "1. Stand with arms at your sides",
            "2. Keep your back straight",
            "3. Raise both arms to shoulder height",
            "4. Continue raising until arms are overhead",
            "5. Lower arms back to starting position",
            "6. Repeat the movement"
        ]
        # Draw simple stick figure for arm raise
        # Down position
        cv2.circle(demo_image, (int(width/4), 150), 20, (0, 255, 0), -1)  # Head
        cv2.line(demo_image, (int(width/4), 170), (int(width/4), 300), (0, 255, 0), 3)  # Body
        cv2.line(demo_image, (int(width/4), 200), (int(width/4) - 30, 260), (0, 255, 0), 3)  # Left arm down
        cv2.line(demo_image, (int(width/4), 200), (int(width/4) + 30, 260), (0, 255, 0), 3)  # Right arm down
        cv2.line(demo_image, (int(width/4), 300), (int(width/4) - 40, 400), (0, 255, 0), 3)  # Left leg
        cv2.line(demo_image, (int(width/4), 300), (int(width/4) + 40, 400), (0, 255, 0), 3)  # Right leg
        
        # Up position
        cv2.circle(demo_image, (int(width*3/4), 150), 20, (0, 255, 0), -1)  # Head
        cv2.line(demo_image, (int(width*3/4), 170), (int(width*3/4), 300), (0, 255, 0), 3)  # Body
        cv2.line(demo_image, (int(width*3/4), 200), (int(width*3/4) - 40, 120), (0, 255, 0), 3)  # Left arm up
        cv2.line(demo_image, (int(width*3/4), 200), (int(width*3/4) + 40, 120), (0, 255, 0), 3)  # Right arm up
        cv2.line(demo_image, (int(width*3/4), 300), (int(width*3/4) - 40, 400), (0, 255, 0), 3)  # Left leg
        cv2.line(demo_image, (int(width*3/4), 300), (int(width*3/4) + 40, 400), (0, 255, 0), 3)  # Right leg
        
        # Add labels
        cv2.putText(demo_image, "Starting Position", (int(width/4) - 80, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(demo_image, "Raised Position", (int(width*3/4) - 80, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    
    elif exercise_type == 'lunge':
        instructions = [
            "1. Stand with feet hip-width apart",
            "2. Step forward with one leg",
            "3. Lower your body until both knees are bent at 90°",
            "4. Keep front knee aligned with ankle",
            "5. Push back to starting position",
            "6. Repeat with alternate legs"
        ]
        # Draw simple stick figure for lunge
        # Standing position
        cv2.circle(demo_image, (int(width/4), 150), 20, (0, 255, 0), -1)  # Head
        cv2.line(demo_image, (int(width/4), 170), (int(width/4), 300), (0, 255, 0), 3)  # Body
        cv2.line(demo_image, (int(width/4), 200), (int(width/4) - 50, 220), (0, 255, 0), 3)  # Left arm
        cv2.line(demo_image, (int(width/4), 200), (int(width/4) + 50, 220), (0, 255, 0), 3)  # Right arm
        cv2.line(demo_image, (int(width/4), 300), (int(width/4) - 20, 400), (0, 255, 0), 3)  # Left leg
        cv2.line(demo_image, (int(width/4), 300), (int(width/4) + 20, 400), (0, 255, 0), 3)  # Right leg
        
        # Lunge position
        cv2.circle(demo_image, (int(width*3/4), 180), 20, (0, 255, 0), -1)  # Head
        cv2.line(demo_image, (int(width*3/4), 200), (int(width*3/4), 280), (0, 255, 0), 3)  # Body
        cv2.line(demo_image, (int(width*3/4), 230), (int(width*3/4) - 50, 250), (0, 255, 0), 3)  # Left arm
        cv2.line(demo_image, (int(width*3/4), 230), (int(width*3/4) + 50, 250), (0, 255, 0), 3)  # Right arm
        cv2.line(demo_image, (int(width*3/4), 280), (int(width*3/4) - 80, 350), (0, 255, 0), 3)  # Left leg back
        cv2.line(demo_image, (int(width*3/4), 280), (int(width*3/4) + 80, 350), (0, 255, 0), 3)  # Right leg forward
        
        # Add labels
        cv2.putText(demo_image, "Starting Position", (int(width/4) - 80, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(demo_image, "Lunge Position", (int(width*3/4) - 70, 400), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    
    elif exercise_type == 'plank':
        instructions = [
            "1. Start in forearm plank position",
            "2. Keep your body in a straight line",
            "3. Engage your core muscles",
            "4. Keep elbows directly under shoulders",
            "5. Hold the position",
            "6. Maintain proper form throughout"
        ]
        # Draw simple stick figure for plank
        cv2.circle(demo_image, (int(width/2), 200), 15, (0, 255, 0), -1)  # Head
        cv2.line(demo_image, (int(width/2), 215), (int(width/2), 300), (0, 255, 0), 3)  # Body
        cv2.line(demo_image, (int(width/2) - 60, 230), (int(width/2), 230), (0, 255, 0), 3)  # Left arm
        cv2.line(demo_image, (int(width/2), 230), (int(width/2) + 60, 230), (0, 255, 0), 3)  # Right arm
        cv2.line(demo_image, (int(width/2), 300), (int(width/2) - 100, 300), (0, 255, 0), 3)  # Left leg
        cv2.line(demo_image, (int(width/2), 300), (int(width/2) + 100, 300), (0, 255, 0), 3)  # Right leg
        
        # Add label
        cv2.putText(demo_image, "Plank Position", (int(width/2) - 70, 350), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Add instructions to the image
    y_pos = height - 250
    for instruction in instructions:
        cv2.putText(demo_image, instruction, (50, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y_pos += 30
    
    # Add prompt to continue
    cv2.putText(demo_image, "Press SPACE to start exercise tracking", (int(width/2) - 200, height - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    
    return demo_image

def fitness_tracking(exercise_type, source=0, duration=60, output_dir=None, model_complexity=2):
    """
    Track fitness exercises using pose detection
    
    Args:
        exercise_type: Type of exercise to count ('squat', 'pushup', 'arm_raise', 'lunge', 'plank')
        source: Camera index or video file path
        duration: Duration to track in seconds (for webcam)
        output_dir: Directory to save results (optional)
        model_complexity: MediaPipe model complexity (0=fastest, 2=most accurate)
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
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        video_path = os.path.join(output_dir, f"{exercise_type}_tracking.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    else:
        video_writer = None
    
    # Show exercise demonstration first
    demo_image = show_exercise_demo(exercise_type, width, height)
    cv2.imshow(f'{exercise_type.capitalize()} Demonstration', demo_image)
    print(f"Showing {exercise_type} demonstration. Press SPACE to start tracking.")
    
    # Wait for user to press space to continue
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # Space key
            break
        elif key == 27:  # ESC key
            cv2.destroyAllWindows()
            cap.release()
            return
    
    cv2.destroyWindow(f'{exercise_type.capitalize()} Demonstration')
    
    # Initialize MediaPipe Pose with specified model complexity for better accuracy
    with mp_pose.Pose(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        model_complexity=model_complexity
    ) as pose:
        
        # Initialize exercise counter
        counter = ExerciseCounter(exercise_type, smoothing_window=10, confidence_threshold=0.6)
        
        # Initialize variables
        start_time = time.time()
        exercise_start_time = time.time()
        frame_count = 0
        
        # Initialize performance metrics
        processing_times = []
        
        # Initialize UI elements
        quality_color = (255, 255, 255)  # Default white
        
        print(f"Starting {exercise_type} tracking. Press 'q' to quit.")
        print("Calibration in progress... Please perform the exercise slowly.")
        
        # Main loop
        while cap.isOpened():
            loop_start = time.time()
            
            # Read frame
            success, image = cap.read()
            if not success:
                break
            
            # Check if we've reached the duration limit for webcam
            if is_webcam and duration and (time.time() - exercise_start_time) > duration:
                print(f"Reached duration limit of {duration} seconds")
                break
            
            # Flip the image horizontally for a selfie-view display if using webcam
            if is_webcam:
                image = cv2.flip(image, 1)
            
            # Create a copy for display
            display_image = image.copy()
            
            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image and detect pose
            process_start = time.time()
            results = pose.process(image_rgb)
            process_time = time.time() - process_start
            processing_times.append(process_time)
            
            # Draw pose landmarks on the image
            if results.pose_landmarks:
                # Extract landmark confidence values
                landmark_confidence = {}
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    landmark_confidence[list(mp_pose.PoseLandmark)[idx].name.lower()] = landmark.visibility
                
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    display_image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # Extract landmarks
                landmarks_dict = extract_landmarks(results)
                
                # Update exercise counter with confidence values
                counter_result = counter.update(landmarks_dict, landmark_confidence)
                
                # Extract results
                rep_count = counter_result['rep_count']
                state = counter_result['state']
                primary_angle = counter_result['primary_angle']
                secondary_angle = counter_result['secondary_angle']
                form_feedback = counter_result['form_feedback']
                rep_quality = counter_result['rep_quality']
                calibrated = counter_result['calibrated']
                
                # Set quality color based on rep quality
                if rep_quality is not None:
                    if rep_quality >= 80:
                        quality_color = (0, 255, 0)  # Green for good
                    elif rep_quality >= 50:
                        quality_color = (0, 255, 255)  # Yellow for medium
                    else:
                        quality_color = (0, 0, 255)  # Red for poor
                
                # Display calibration status
                if not calibrated:
                    cv2.putText(display_image, "Calibrating... Please perform the exercise", (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
                    
                    # Draw progress bar for calibration
                    progress = min(1.0, counter.calibration_frames / counter.calibration_max_frames)
                    bar_width = int(width * 0.6)
                    bar_height = 20
                    bar_x = int(width * 0.2)
                    bar_y = 80
                    
                    # Draw background
                    cv2.rectangle(display_image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
                    # Draw progress
                    cv2.rectangle(display_image, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)
                    # Draw border
                    cv2.rectangle(display_image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
                else:
                    # Display rep count with large font
                    cv2.putText(display_image, f"Reps: {rep_count}", (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    # Display state
                    state_color = (0, 255, 0) if state == "up" else (0, 165, 255)
                    cv2.putText(display_image, f"State: {state.upper()}", (20, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2, cv2.LINE_AA)
                    
                    # Display angles
                    if primary_angle is not None:
                        cv2.putText(display_image, f"{counter.joint_name.capitalize()} Angle: {primary_angle:.1f}°", (20, 130), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    if secondary_angle is not None:
                        cv2.putText(display_image, f"{counter.secondary_joint_name.capitalize()} Angle: {secondary_angle:.1f}°", (20, 170), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Display form feedback
                    if form_feedback:
                        cv2.putText(display_image, f"Feedback: {form_feedback}", (20, 210), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
                    
                    # Display last rep quality if available
                    if rep_quality is not None:
                        cv2.putText(display_image, f"Last Rep Quality: {rep_quality}%", (20, 250), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, quality_color, 2, cv2.LINE_AA)
                    
                    # Draw quality meter for the last rep
                    if rep_quality is not None:
                        meter_width = 200
                        meter_height = 20
                        meter_x = 20
                        meter_y = 280
                        
                        # Draw background
                        cv2.rectangle(display_image, (meter_x, meter_y), (meter_x + meter_width, meter_y + meter_height), (100, 100, 100), -1)
                        # Draw quality level
                        quality_width = int(meter_width * (rep_quality / 100))
                        cv2.rectangle(display_image, (meter_x, meter_y), (meter_x + quality_width, meter_y + meter_height), quality_color, -1)
                        # Draw border
                        cv2.rectangle(display_image, (meter_x, meter_y), (meter_x + meter_width, meter_y + meter_height), (255, 255, 255), 2)
            
            # Calculate and display FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                current_fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()
            else:
                current_fps = fps
            
            cv2.putText(display_image, f"FPS: {current_fps:.1f}", (width - 150, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Display exercise instructions
            if exercise_type == 'squat':
                instructions = "Squat down until knees are bent, then stand up"
            elif exercise_type == 'pushup':
                instructions = "Lower body until arms are bent, then push up"
            elif exercise_type == 'arm_raise':
                instructions = "Raise arms up, then lower them down"
            elif exercise_type == 'lunge':
                instructions = "Step forward, lower knee, then return"
            elif exercise_type == 'plank':
                instructions = "Hold position with straight back"
            else:
                instructions = "Perform the exercise with good form"
            
            cv2.putText(display_image, instructions, (20, height - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            
            # Display the image
            cv2.imshow(f'{exercise_type.capitalize()} Tracker', display_image)
            
            # Write frame to video if output directory is specified
            if video_writer:
                video_writer.write(display_image)
            
            # Calculate loop time for performance monitoring
            loop_time = time.time() - loop_start
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release resources
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    
    # Get exercise statistics
    stats = counter.get_statistics()
    
    # Print exercise summary
    print("\nExercise Summary:")
    print(f"Total {exercise_type} repetitions: {stats['rep_count']}")
    if stats['avg_quality'] > 0:
        print(f"Average rep quality: {stats['avg_quality']:.1f}%")
    if stats['avg_duration'] > 0:
        print(f"Average rep duration: {stats['avg_duration']:.2f} seconds")
    if stats['range_of_motion'] > 0:
        print(f"Range of motion: {stats['range_of_motion']:.1f} degrees")
    
    # Calculate and print performance metrics
    if processing_times:
        avg_process_time = sum(processing_times) / len(processing_times)
        print(f"\nPerformance Metrics:")
        print(f"Average frame processing time: {avg_process_time*1000:.2f} ms")
        print(f"Estimated maximum FPS: {1/avg_process_time:.1f}")
    
    # Save statistics to file if output directory is specified
    if output_dir:
        stats_file = os.path.join(output_dir, f"{exercise_type}_stats.txt")
        with open(stats_file, 'w') as f:
            f.write(f"Exercise: {exercise_type}\n")
            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total repetitions: {stats['rep_count']}\n")
            f.write(f"Average rep quality: {stats['avg_quality']:.1f}%\n")
            f.write(f"Average rep duration: {stats['avg_duration']:.2f} seconds\n")
            f.write(f"Range of motion: {stats['range_of_motion']:.1f} degrees\n")
        
        print(f"\nStatistics saved to {stats_file}")

def show_exercise_menu(width=800, height=600):
    """
    Display a visual menu for exercise selection
    
    Args:
        width: Width of the menu window
        height: Height of the menu window
        
    Returns:
        selected_exercise: The selected exercise type
    """
    # Define available exercises with icons and names
    exercises = [
        {'id': 'squat', 'name': 'Squat', 'icon': '🏋️'},
        {'id': 'pushup', 'name': 'Push-up', 'icon': '💪'},
        {'id': 'arm_raise', 'name': 'Arm Raise', 'icon': '🙌'},
        {'id': 'lunge', 'name': 'Lunge', 'icon': '🚶'},
        {'id': 'plank', 'name': 'Plank', 'icon': '🧘'},
        {'id': 'jumping_jack', 'name': 'Jumping Jack', 'icon': '⭐'},
        {'id': 'bicycle_crunch', 'name': 'Bicycle Crunch', 'icon': '🚲'},
        {'id': 'mountain_climber', 'name': 'Mountain Climber', 'icon': '🏔️'},
        {'id': 'side_plank', 'name': 'Side Plank', 'icon': '🔄'},
        {'id': 'burpee', 'name': 'Burpee', 'icon': '🔥'},
        {'id': 'tricep_dip', 'name': 'Tricep Dip', 'icon': '👊'},
        {'id': 'calf_raise', 'name': 'Calf Raise', 'icon': '🦵'},
    ]
    
    # Create menu image
    menu_image = np.zeros((height, width, 3), dtype=np.uint8)
    menu_image[:] = (50, 50, 50)  # Dark gray background
    
    # Add title
    title = "FITNESS EXERCISE SELECTION"
    cv2.putText(menu_image, title, (int(width/2) - 200, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Add instructions
    instructions = "Press the number key to select an exercise"
    cv2.putText(menu_image, instructions, (int(width/2) - 200, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Calculate grid layout
    cols = 3
    rows = (len(exercises) + cols - 1) // cols
    cell_width = width // cols
    cell_height = (height - 120) // rows
    
    # Draw exercise options
    for i, exercise in enumerate(exercises):
        row = i // cols
        col = i % cols
        
        x = col * cell_width + 20
        y = row * cell_height + 150
        
        # Draw selection box
        cv2.rectangle(menu_image, (x - 10, y - 30), (x + cell_width - 30, y + cell_height - 60), (70, 70, 70), -1)
        cv2.rectangle(menu_image, (x - 10, y - 30), (x + cell_width - 30, y + cell_height - 60), (100, 100, 100), 2)
        
        # Add number
        cv2.putText(menu_image, f"{i+1}", (x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Add icon
        cv2.putText(menu_image, exercise['icon'], (x + 40, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Add name
        cv2.putText(menu_image, exercise['name'], (x, y + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Display menu
    cv2.imshow("Exercise Selection Menu", menu_image)
    
    # Wait for key press
    selected_index = None
    while selected_index is None:
        key = cv2.waitKey(0) & 0xFF
        
        # Check for number keys 1-9 and 0 (for 10)
        if ord('1') <= key <= ord('9'):
            selected_index = key - ord('1')
        elif key == ord('0'):
            selected_index = 9
        elif key == ord('-'):
            selected_index = 10
        elif key == ord('='):
            selected_index = 11
        elif key == 27:  # ESC key
            cv2.destroyAllWindows()
            return None
        
        # Validate selection
        if selected_index is not None and selected_index < len(exercises):
            selected_exercise = exercises[selected_index]['id']
        else:
            selected_index = None
    
    cv2.destroyAllWindows()
    return selected_exercise


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fitness tracking using pose detection')
    parser.add_argument('--exercise', '-e', type=str, required=False, 
                        choices=['squat', 'pushup', 'arm_raise', 'lunge', 'plank', 
                                'jumping_jack', 'bicycle_crunch', 'mountain_climber', 
                                'side_plank', 'burpee', 'tricep_dip', 'calf_raise'],
                        help='Type of exercise to track')
    parser.add_argument('--source', '-s', type=str, default="0",
                        help='Camera index (e.g., 0) or video file path')
    parser.add_argument('--duration', '-d', type=int, default=60,
                        help='Duration to track in seconds (for webcam)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Directory to save results (optional)')
    parser.add_argument('--model-complexity', '-m', type=int, default=2, choices=[0, 1, 2],
                        help='MediaPipe model complexity (0=fastest, 2=most accurate)')
    parser.add_argument('--menu', action='store_true', help='Show exercise selection menu')
    args = parser.parse_args()
    
    # Show menu if requested or if no exercise specified
    exercise_type = args.exercise
    if args.menu or exercise_type is None:
        exercise_type = show_exercise_menu()
        if exercise_type is None:
            print("No exercise selected. Exiting.")
            return
    
    # Convert source to int if it's a number (camera index)
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        print(f"Results will be saved to: {args.output}")
    
    # Track fitness exercise
    fitness_tracking(
        exercise_type=exercise_type,
        source=source,
        duration=args.duration,
        output_dir=args.output,
        model_complexity=args.model_complexity
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nFitness tracking interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        # Release resources in case of error
        cv2.destroyAllWindows()
