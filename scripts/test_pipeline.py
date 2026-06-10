import os
import subprocess
import sys

# Define paths dynamically based on the script location
current_dir = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR = os.path.join(current_dir, "deep_learning_4")

def run_script(script_name, env):
    print(f"\n{'-'*30}")
    print(f"▶ RUNNING: {script_name}")
    print(f"{'-'*30}")
    
    script_path = os.path.join(TARGET_DIR, script_name)

    if not os.path.exists(script_path):
        print(f"\n❌ ERROR: Cannot find '{script_path}'.")
        sys.exit(1)
    
    try:
        # Run the script sequentially and wait for it to finish
        subprocess.run([sys.executable, script_path], env=env, check=True)
    except subprocess.CalledProcessError:
        print(f"\n❌ ERROR: Pipeline crashed during {script_name}. Halting execution.")
        sys.exit(1)

def main():
    print("Starting Sequential Execution Pipeline (Optimal Parameters)...")
    
    # Define the exact execution order
    pipeline_stages = [
        "train_gru.py",
        "train_lstm.py",
        "train_cnn.py",
        "train_dnn.py",
        "error_analysis_all.py",
        "evaluate_system.py"
    ]

    # Grab base environment variables
    run_env = os.environ.copy()
    
    # # OPTIONAL: Force your "optimal" parameters here so the MLConfig picks them up automatically
    # run_env['ML_WINDOW_SIZE'] = '16'
    # run_env['ML_USE_KDE'] = 'True'
    # run_env['ML_CONSTRAINT_TYPE'] = 'AR'
    # run_env['ML_SEED'] = '42'
    # run_env['ML_EXPERIMENT_NAME'] = 'optimal_local_test'

    # Run the scripts one by one
    for stage in pipeline_stages:
        run_script(stage, run_env)

    print("\n✅ ALL SCRIPTS EXECUTED SUCCESSFULLY!")

if __name__ == "__main__":
    main()