import unittest
import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pose_utils import calculate_angle, extract_features

class TestPoseUtils(unittest.TestCase):
    """
    Unit tests for the pose_utils module
    """
    
    def test_calculate_angle(self):
        """
        Test the calculate_angle function
        """
        # Test with a right angle
        a = [0, 0]
        b = [0, 0]
        c = [1, 0]
        angle = calculate_angle(a, b, c)
        self.assertAlmostEqual(angle, 0.0, places=1)
        
        # Test with a 90-degree angle
        a = [0, 0]
        b = [1, 0]
        c = [1, 1]
        angle = calculate_angle(a, b, c)
        self.assertAlmostEqual(angle, 90.0, places=1)
        
        # Test with a 180-degree angle
        a = [0, 0]
        b = [1, 0]
        c = [2, 0]
        angle = calculate_angle(a, b, c)
        self.assertAlmostEqual(angle, 180.0, places=1)
    
    def test_extract_features(self):
        """
        Test the extract_features function
        """
        # Test with empty landmarks
        landmarks_dict = {}
        features = extract_features(landmarks_dict)
        self.assertEqual(features, {})
        
        # Test with minimal landmarks
        landmarks_dict = {
            'left_shoulder': [0.1, 0.2, 0.3, 0.9],
            'left_elbow': [0.2, 0.3, 0.4, 0.9],
            'left_wrist': [0.3, 0.4, 0.5, 0.9]
        }
        features = extract_features(landmarks_dict)
        self.assertIn('left_elbow_angle', features)
        
        # Test with more landmarks
        landmarks_dict = {
            'left_shoulder': [0.1, 0.2, 0.3, 0.9],
            'left_elbow': [0.2, 0.3, 0.4, 0.9],
            'left_wrist': [0.3, 0.4, 0.5, 0.9],
            'right_shoulder': [0.9, 0.2, 0.3, 0.9],
            'right_elbow': [0.8, 0.3, 0.4, 0.9],
            'right_wrist': [0.7, 0.4, 0.5, 0.9]
        }
        features = extract_features(landmarks_dict)
        self.assertIn('left_elbow_angle', features)
        self.assertIn('right_elbow_angle', features)
        self.assertIn('left_wrist_to_shoulder_y', features)
        self.assertIn('right_wrist_to_shoulder_y', features)

if __name__ == '__main__':
    unittest.main()
