# Pose Estimation for Human Activity Recognition

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
│       └── synthetic_data.csv # Generated synthetic data
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

## Code Organization

### Core Modules

1. **pose_utils.py**
   - Landmark extraction
   - Feature calculation
   - Angle calculation
   - Visualization utilities

2. **activity_classifier.py**
   - ActivityClassifier class
   - Rule-based classification
   - Temporal smoothing
   - Confidence calculation

3. **pose_detection.py**
   - Real-time pose detection
   - Camera integration
   - Frame processing
   - Activity visualization

4. **model_inference.py**
   - Model loading
   - Feature preparation
   - Activity prediction
   - Confidence calculation

### Scripts

1. **generate_synthetic_data.py**
   - Generate synthetic pose data
   - Save to CSV format
   - Command-line interface

2. **train_model.py**
   - Load training data
   - Feature extraction
   - Model training
   - Model evaluation
   - Model saving

3. **data_collection.py**
   - Collect pose data from webcam
   - Save landmarks to CSV
   - Command-line interface

4. **run_pipeline.py**
   - End-to-end pipeline
   - Data generation
   - Model training
   - Real-time inference

## Installation and Setup

```bash
# Clone the repository
git clone https://github.com/username/pose-estimation-activity-recognition.git
cd pose-estimation-activity-recognition

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Usage

### Running the Pipeline

```bash
# Run the complete pipeline
python scripts/run_pipeline.py

# Generate synthetic data
python scripts/generate_synthetic_data.py --output data/synthetic/synthetic_data.csv --samples 500

# Train a model
python scripts/train_model.py data/synthetic/synthetic_data.csv --output models

# Run real-time inference
python -m src.model_inference --model models
```

### Using the Notebook

1. Open the notebook in Jupyter or Google Colab:
   ```bash
   jupyter notebook notebooks/Human_Activity_Recognition.ipynb
   ```

2. Follow the instructions in the notebook to:
   - Install dependencies
   - Process images
   - Perform real-time activity recognition
