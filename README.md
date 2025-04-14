# Pose Estimation for Human Activity Recognition

A comprehensive real-time human pose estimation system for activity recognition using MediaPipe and OpenCV.

## Features

### Core Features
- Real-time pose detection using webcam
- Keypoint visualization with angle measurements
- Activity recognition (standing, sitting, walking, raising arms)
- Machine learning-based pose classification
- Synthetic data generation for training
- Comprehensive fitness tracking

## Repository Structure

```
pose-estimation-activity-recognition/
│
├── README.md                  # Project documentation
├── requirements.txt           # Dependencies
├── setup.py                   # Package installation
├── .gitignore                 # Git ignore file
│
├── src/                       # Source code
│   ├── __init__.py            # Package initialization
│   ├── pose_utils.py          # Core utilities for pose processing
│   ├── activity_classifier.py # Activity classification logic
│   ├── pose_detection.py      # Real-time pose detection
│   └── model_inference.py     # Using trained models for inference
│
├── data/                      # Data directory
│   ├── raw/                   # Raw data
│   ├── processed/             # Processed data
│   └── synthetic/             # Synthetic data
│
├── models/                    # Trained models
│   ├── activity_model.pkl     # Trained activity recognition model
│   ├── feature_scaler.pkl     # Feature scaler
│   └── feature_names.pkl      # Feature names
│
├── notebooks/                 # Jupyter notebooks
│   └── Human_Activity_Recognition.ipynb  # Main notebook
│
├── scripts/                   # Utility scripts
│   ├── generate_synthetic_data.py  # Generate synthetic data
│   ├── train_model.py              # Train activity recognition model
│   ├── data_collection.py          # Collect pose data
│   └── run_pipeline.py             # Run the complete pipeline
│
└── tests/                     # Unit tests
    ├── __init__.py            # Test package initialization
    ├── test_pose_utils.py     # Tests for pose utilities
    └── test_activity_classifier.py  # Tests for activity classifier
```

## Requirements

- Python 3.8+
- Webcam
- Required packages (see requirements.txt)

## Installation

1. Clone this repository
2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
4. Install the package in development mode:
   ```
   pip install -e .
   ```

## Quick Start

### Run the Complete Pipeline

The easiest way to get started is to run the complete pipeline:

```
python scripts/run_pipeline.py
```

This will:
1. Generate synthetic training data
2. Train an activity recognition model
3. Run real-time inference

### Data Collection and Training

To collect your own training data:

```
python scripts/data_collection.py --output data/raw --duration 30
```

To generate synthetic data instead:

```
python scripts/generate_synthetic_data.py --output data/synthetic/synthetic_data.csv --samples 500
```

To train a model on collected or synthetic data:

```
python scripts/train_model.py data/synthetic/synthetic_data.csv --output models
```

### Activity Recognition

For real-time activity recognition:

```
python -m src.pose_detection
```

For model-based activity recognition:

```
python -m src.model_inference --model models
```

### Using the Notebook

Open the notebook in Jupyter or Google Colab:

```
jupyter notebook notebooks/Human_Activity_Recognition.ipynb
```

Follow the instructions in the notebook to:
- Install dependencies
- Process images
- Perform real-time activity recognition

## How It Works

1. **Pose Detection**: MediaPipe's pose estimation model detects 33 key body landmarks in each frame
2. **Feature Extraction**: Joint angles and relative positions are calculated from landmarks
3. **Activity Recognition**: Features are used to classify activities using either:
   - Rule-based classification (activity_classifier.py)
   - Machine learning model (trained with train_model.py)

## Testing

Run the unit tests:

```
python -m unittest discover tests
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Google MediaPipe team for the pose estimation models
- OpenCV community for computer vision tools
- Contributors to the scikit-learn library
