import unittest
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.activity_classifier import ActivityClassifier

class TestActivityClassifier(unittest.TestCase):
    """
    Unit tests for the ActivityClassifier class
    """
    
    def setUp(self):
        """
        Set up the test case
        """
        self.classifier = ActivityClassifier(history_size=10)
    
    def test_initialization(self):
        """
        Test the initialization of the ActivityClassifier
        """
        self.assertEqual(self.classifier.current_activity, "Unknown")
        self.assertEqual(len(self.classifier.history), 0)
        self.assertEqual(self.classifier.stability_threshold, 5)
        self.assertEqual(set(self.classifier.activity_counter.keys()), 
                         {"Unknown", "Standing", "Sitting", "Walking/Moving", "Raising Arms"})
    
    def test_empty_features(self):
        """
        Test classification with empty features
        """
        activity = self.classifier.update({})
        self.assertEqual(activity, "Unknown")
    
    def test_standing_classification(self):
        """
        Test classification of standing activity
        """
        # Create features for standing
        features = {
            'left_knee_angle': 170,
            'right_knee_angle': 170,
            'hip_ankle_ratio': 1.5,
            'left_wrist_to_shoulder_y': 0.0,
            'right_wrist_to_shoulder_y': 0.0
        }
        
        # Update classifier multiple times to exceed stability threshold
        for _ in range(6):
            activity = self.classifier.update(features)
        
        self.assertEqual(activity, "Standing")
    
    def test_sitting_classification(self):
        """
        Test classification of sitting activity
        """
        # Create features for sitting
        features = {
            'left_knee_angle': 90,
            'right_knee_angle': 90,
            'hip_ankle_ratio': 1.0,
            'left_wrist_to_shoulder_y': 0.0,
            'right_wrist_to_shoulder_y': 0.0
        }
        
        # Update classifier multiple times to exceed stability threshold
        for _ in range(6):
            activity = self.classifier.update(features)
        
        self.assertEqual(activity, "Sitting")
    
    def test_raising_arms_classification(self):
        """
        Test classification of raising arms activity
        """
        # Create features for raising arms
        features = {
            'left_knee_angle': 170,
            'right_knee_angle': 170,
            'hip_ankle_ratio': 1.5,
            'left_wrist_to_shoulder_y': 0.2,
            'right_wrist_to_shoulder_y': 0.2
        }
        
        # Update classifier multiple times to exceed stability threshold
        for _ in range(6):
            activity = self.classifier.update(features)
        
        self.assertEqual(activity, "Raising Arms")
    
    def test_walking_classification(self):
        """
        Test classification of walking activity
        """
        # Create features for walking
        features = {
            'left_knee_angle': 150,
            'right_knee_angle': 150,
            'hip_ankle_ratio': 1.5,
            'left_wrist_to_shoulder_y': 0.0,
            'right_wrist_to_shoulder_y': 0.0
        }
        
        # Update classifier multiple times to exceed stability threshold
        for _ in range(6):
            activity = self.classifier.update(features)
        
        self.assertEqual(activity, "Walking/Moving")
    
    def test_activity_confidence(self):
        """
        Test getting activity confidence scores
        """
        # Create features for standing
        features = {
            'left_knee_angle': 170,
            'right_knee_angle': 170,
            'hip_ankle_ratio': 1.5,
            'left_wrist_to_shoulder_y': 0.0,
            'right_wrist_to_shoulder_y': 0.0
        }
        
        # Update classifier multiple times
        for _ in range(10):
            self.classifier.update(features)
        
        # Get confidence scores
        confidence = self.classifier.get_activity_confidence()
        
        # Check that confidence scores sum to 1
        self.assertAlmostEqual(sum(confidence.values()), 1.0, places=5)
        
        # Check that standing has the highest confidence
        self.assertEqual(max(confidence, key=confidence.get), "Standing")

if __name__ == '__main__':
    unittest.main()
