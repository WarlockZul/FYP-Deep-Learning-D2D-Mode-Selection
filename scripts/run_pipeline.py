import os
import subprocess
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, ".."))
TARGET_DIR = os.path.join(PROJECT_ROOT, "scripts", "deep_learning_4")

def run_script(script_name, env=None):
    print(f"\n{'-'*40}")
    print(f"▶ RUNNING: {script_name}")
    print(f"{'-'*40}")
    
    script_path = os.path.join(TARGET_DIR, script_name)

    if not os.path.exists(script_path):
        print(f"\n❌ ERROR: Cannot find '{script_path}'.")
        print("Please ensure the file is exactly inside ~/scripts/deep_learning_4/")
        sys.exit(1)
    
    # Use subprocess to run the file through python script instead of typing in terminal
    try:
        subprocess.run([sys.executable, script_path], env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: Pipeline crashed during {script_name}. Halting execution.")
        sys.exit(1)

def main():
    print("Starting Automated 3-Module ML Pipeline...")

    # Scripts that can run at the exact same time
    parallel_training_scripts = [
        "train_gru.py",
        "train_lstm.py",
        "train_cnn.py",
        "train_dnn.py"
    ]
    
    # Scripts that must run one after the other
    sequential_eval_scripts = [
        "error_analysis_all.py",
        "evaluate_system.py"
    ]

    # Ablation Study: Loop through different window sizes, KDE options, and constraint types
    window_sizes = [64]
    kde_options = ['False'] 
    constraint_types = ['AR', 'PCR']
    seeds = [21, 61, 123]  
    
    # Calculate total experiments for progress tracking
    total_experiments = len(window_sizes) * len(kde_options) * len(constraint_types) * len(seeds)
    current_experiment = 1

    # Grab base environment variables
    base_env = os.environ.copy()
    
    # NEW: Force the pipeline to strictly use the proposal dataset
    base_env['ML_DATASETS'] = 'preprocessed_proposal'
    os.environ['ML_DATASETS'] = 'preprocessed_proposal'

    for w in window_sizes:
        for kde in kde_options:
            for constraint in constraint_types:
                for seed in seeds:
                    
                    # Create a dynamic, readable folder name that includes the seed
                    exp_name = f"ablation_win{w}_kde{kde}_{constraint}_seed{seed}"
                    
                    print(f"\n\n{'#'*30}")
                    print(f"🌟 EXPERIMENT {current_experiment}/{total_experiments}: {exp_name.upper()}")
                    print(f"{'#'*30}")
                    
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
                    print(f"\n🚀 Launching 4 Deep Learning Models concurrently for Proposal Dataset...")
                    active_processes = []
                    
                    for script in parallel_training_scripts:
                        script_path = os.path.join(TARGET_DIR, script)
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
                        run_script(stage, env=run_env)
                    
                    current_experiment += 1
            
    print("\n✅ ALL EXPERIMENTS COMPLETE! \nCheck your ~/results/ and ~/models/ folders.")

if __name__ == "__main__":
    main()