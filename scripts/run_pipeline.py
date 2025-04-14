import os
import subprocess
import argparse
import time

def run_command(cmd, description):
    """
    Run a command and print its output
    
    Args:
        cmd: Command to run
        description: Description of the command
    """
    print(f"\n{'='*50}")
    print(f"{description}")
    print(f"{'='*50}")
    print(f"Running command: {' '.join(cmd)}")
    
    # Run command
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed_time = time.time() - start_time
    
    # Print output
    print(f"\nCommand completed in {elapsed_time:.2f} seconds")
    print(f"Return code: {result.returncode}")
    
    if result.stdout:
        print("\nStandard output:")
        print(result.stdout)
    
    if result.stderr:
        print("\nStandard error:")
        print(result.stderr)
    
    return result.returncode == 0

def run_pipeline(use_synthetic_data=True, data_dir="data", model_dir="models", samples=500):
    """
    Run the entire pipeline: generate data, train model, and evaluate
    
    Args:
        use_synthetic_data: Whether to use synthetic data or collect real data
        data_dir: Directory to store data
        model_dir: Directory to store models
        samples: Number of samples to generate per activity (for synthetic data)
    """
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Step 1: Generate or collect data
    if use_synthetic_data:
        # Generate synthetic data
        data_file = os.path.join(data_dir, "synthetic_data.csv")
        cmd = [
            "python", "generate_synthetic_data.py",
            "--output", data_file,
            "--samples", str(samples),
            "--noise", "0.05"
        ]
        
        success = run_command(cmd, "Step 1: Generating synthetic data")
        if not success:
            print("Failed to generate synthetic data. Exiting.")
            return False
    else:
        # Collect real data
        cmd = [
            "python", "collect_sample_data.py",
            "--output", data_dir,
            "--duration", "30",
            "--webcam", "0"
        ]
        
        print("\nWARNING: You will need to perform each activity when prompted.")
        input("Press Enter to continue...")
        
        success = run_command(cmd, "Step 1: Collecting real data")
        if not success:
            print("Failed to collect real data. Exiting.")
            return False
        
        data_file = os.path.join(data_dir, "collected_data.csv")
    
    # Step 2: Train model
    cmd = [
        "python", "train_model.py",
        data_file,
        "--output", model_dir
    ]
    
    success = run_command(cmd, "Step 2: Training activity recognition model")
    if not success:
        print("Failed to train model. Exiting.")
        return False
    
    # Step 3: Test model with real-time inference
    cmd = [
        "python", "model_inference.py",
        "--model", os.path.join(model_dir, "activity_model.pkl"),
        "--scaler", os.path.join(model_dir, "feature_scaler.pkl"),
        "--features", os.path.join(model_dir, "feature_names.pkl")
    ]
    
    print("\nStep 3: Testing model with real-time inference")
    print("This will open a window for real-time activity recognition.")
    print("Press 'q' to quit the demo when you're done.")
    input("Press Enter to continue...")
    
    success = run_command(cmd, "Running real-time inference")
    if not success:
        print("Failed to run real-time inference. Exiting.")
        return False
    
    print("\nPipeline completed successfully!")
    return True

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the entire human activity recognition pipeline')
    parser.add_argument('--real-data', action='store_true',
                        help='Collect real data instead of using synthetic data')
    parser.add_argument('--data-dir', type=str, default="data",
                        help='Directory to store data (default: data)')
    parser.add_argument('--model-dir', type=str, default="models",
                        help='Directory to store models (default: models)')
    parser.add_argument('--samples', type=int, default=500,
                        help='Number of samples to generate per activity (default: 500)')
    args = parser.parse_args()
    
    # Run pipeline
    run_pipeline(
        use_synthetic_data=not args.real_data,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        samples=args.samples
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
