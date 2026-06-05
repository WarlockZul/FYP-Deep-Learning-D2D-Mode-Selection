import os
import subprocess
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, ".."))
TARGET_DIR = os.path.join(PROJECT_ROOT, "scripts", "deep_learning_4")

def run_script(script_name):
    print(f"\n{'-'*40}")
    print(f"▶ RUNNING: {script_name}")
    print(f"{'-'*40}")
    
    script_path = os.path.join(TARGET_DIR, script_name)

    if not os.path.exists(script_path):
        print(f"\n❌ ERROR: Cannot find '{script_path}'.")
        print("Please ensure the file is exactly inside ~/scripts/deep_learning_4/")
        sys.exit(1)
    
    # Use subprocess to run the file through python script instead of typing in terminal
    # Able to catch errors and halt the pipeline if any script fails
    try:
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: Pipeline crashed during {script_name}. Halting execution.")
        sys.exit(1)

def main():
    print("Starting Automated 3-Module ML Pipeline...")
    
    # # Define the exact execution order (SINR Prediction -> Error Analysis -> System Evaluation)
    # pipeline_stages = [
    #     "train_gru.py",
    #     "train_lstm.py",
    #     "train_cnn.py",
    #     "train_dnn.py",
    #     "error_analysis_all.py",
    #     "evaluate_system.py"
    # ]

    # Scripts that can run at the exact same time
    parallel_training_scripts = [
        "train_gru.py",
        "train_lstm.py",
        "train_cnn.py",
        "train_dnn.py"
    ]
    
    # Scripts that must run one after the other (they depend on the models being finished)
    sequential_eval_scripts = [
        "error_analysis_all.py",
        "evaluate_system.py"
    ]

    # Ablation Study: Loop through different window sizes, KDE options, and constraint types
    window_sizes = [16, 32, 64]
    kde_options = ['True', 'False'] 
    constraint_types = ['AR', 'PCR']
    seeds = [21, 99, 123]  
    
    # Calculate total experiments for progress tracking
    total_experiments = len(window_sizes) * len(kde_options) * len(constraint_types) * len(seeds)
    current_experiment = 1

    # Grab base environment variables
    base_env = os.environ.copy()

    for w in window_sizes:
        for kde in kde_options:
            for constraint in constraint_types:
                for seed in seeds:
                    
                    # Create a dynamic, readable folder name that includes the seed
                    # Example: "ablation_win16_kdeTrue_AR_seed42"
                    exp_name = f"ablation_win{w}_kde{kde}_{constraint}_seed{seed}"
                    
                    print(f"\n\n{'#'*50}")
                    print(f"🌟 EXPERIMENT {current_experiment}/{total_experiments}: {exp_name.upper()}")
                    print(f"{'#'*50}")
                    
                    # Broadcast the hyperparameters AND the seed via environment variables
                    os.environ['ML_WINDOW_SIZE'] = str(w)
                    os.environ['ML_USE_KDE'] = kde
                    os.environ['ML_CONSTRAINT_TYPE'] = constraint
                    os.environ['ML_SEED'] = str(seed)
                    os.environ['ML_EXPERIMENT_NAME'] = exp_name
                    
                    # # Run the entire AI pipeline for this specific combination
                    # for stage in pipeline_stages:
                    #     run_script(stage)

                    # Create a specific environment for this run
                    run_env = base_env.copy()
                    run_env['ML_WINDOW_SIZE'] = str(w)
                    run_env['ML_USE_KDE'] = kde
                    run_env['ML_CONSTRAINT_TYPE'] = constraint
                    run_env['ML_SEED'] = str(seed)
                    run_env['ML_EXPERIMENT_NAME'] = exp_name
                    
                    # ==========================================
                    # PHASE 1: PARALLEL TRAINING
                    # ==========================================
                    print(f"\n🚀 Launching 4 Deep Learning Models concurrently...")
                    active_processes = []
                    
                    for script in parallel_training_scripts:
                        script_path = os.path.join(TARGET_DIR, script)
                        # Popen runs the script in the background without waiting
                        p = subprocess.Popen([sys.executable, script_path], env=run_env)
                        active_processes.append((script, p))
                    
                    # Force Python to wait here until all 4 background processes finish
                    for script, p in active_processes:
                        p.wait()
                        if p.returncode != 0:
                            print(f"\n❌ ERROR: {script} crashed during parallel execution. Halting.")
                            sys.exit(1)
                            
                    print(f"\n✅ All 4 models finished training successfully!")

                    # ==========================================
                    # PHASE 2: SEQUENTIAL EVALUATION
                    # ==========================================
                    for stage in sequential_eval_scripts:
                        run_script(stage)
                    
                    current_experiment += 1
            
    print("\n✅ ALL EXPERIMENTS COMPLETE! \nCheck your ~/results/ and ~/models/ folders.")

if __name__ == "__main__":
    main()