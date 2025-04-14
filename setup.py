from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pose-estimation-activity-recognition",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Human activity recognition using pose estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/pose-estimation-activity-recognition",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "mediapipe>=0.8.9",
        "opencv-python>=4.5.3",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
    ],
)
