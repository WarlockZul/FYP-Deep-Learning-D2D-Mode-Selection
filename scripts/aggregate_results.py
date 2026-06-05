import os
import glob
import pandas as pd

# Define paths dynamically based on the script location
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

def aggregate_ablation_results():
    print("Scanning for result files...")
    
    # Target all final_evaluation_metrics.csv files inside any ablation folder
    search_pattern = os.path.join(RESULTS_DIR, "*", "ablation_*", "final_evaluation_metrics.csv")
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        print("❌ No evaluation metrics found. Please ensure the pipeline has completed.")
        return

    all_data = []

    for file_path in csv_files:
        # Extract the dataset folder and experiment name from the directory path
        path_parts = file_path.split(os.sep)
        dataset_folder = path_parts[-3]  # e.g., 'preprocessed_proposal'
        experiment_name = path_parts[-2] # e.g., 'ablation_win16_kdeTrue_AR_seed21'
        
        # Parse the experiment name to extract ablation parameters and seed
        exp_parts = experiment_name.split('_')
        try:
            window_size = int(exp_parts[1].replace('win', ''))
            kde_enabled = exp_parts[2].replace('kde', '')
            constraint = exp_parts[3]
            seed = int(exp_parts[4].replace('seed', ''))
        except Exception as e:
            print(f"Skipping {experiment_name} due to unexpected naming format.")
            continue

        # Load the CSV
        try:
            df = pd.read_csv(file_path)
            
            # FIX 1: Rename the empty first column to 'Model'
            df.rename(columns={df.columns[0]: 'Model'}, inplace=True)
            
            # FIX 2: Strip leading/trailing whitespaces from all column names
            df.columns = df.columns.str.strip()
            
            # Strip whitespaces from the model names (e.g., 'gru      ' -> 'gru')
            df['Model'] = df['Model'].astype(str).str.strip()
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
            
        # Append the parsed parameters as new columns
        df['Dataset'] = dataset_folder
        df['Window_Size'] = window_size
        df['KDE_Enabled'] = kde_enabled
        df['Constraint'] = constraint
        df['Seed'] = seed
        
        all_data.append(df)

    if not all_data:
        print("❌ No valid data could be extracted.")
        return

    # Combine all individual dataframes into one master dataframe
    master_df = pd.concat(all_data, ignore_index=True)
    print(f"Successfully loaded data from {len(csv_files)} experiments.")

    # We only want to group by these parameters
    grouping_columns = ['Dataset', 'Model', 'Window_Size', 'KDE_Enabled', 'Constraint']
    
    # Exact target metrics based on your CSV structure
    target_metrics = [
        'D2D PICP%', 
        'Cell PICP%', 
        'D2D MPIW (dB)', 
        'Cell MPIW (dB)', 
        'Avg Tput (Mbps)'
    ] 
    
    # Filter only the DL models and baselines we actually care about (optional, but keeps it clean)
    master_df = master_df[master_df['Model'].isin(['gru', 'lstm', 'cnn', 'dnn', 'pure_d2d', 'pure_cellular', 'random', 'sinr_threshold'])]

    print("Aggregating mean and standard deviation across seeds...")
    
    # We must convert target metrics to numeric, coercing 'N/A' strings to actual NaNs so math works
    for col in target_metrics:
        master_df[col] = pd.to_numeric(master_df[col], errors='coerce')

    # Group by the ablation parameters and calculate mean and std
    aggregated_df = master_df.groupby(grouping_columns)[target_metrics].agg(['mean', 'std']).reset_index()

    # Create the final clean dataframe for the thesis table
    thesis_table = pd.DataFrame()
    for col in grouping_columns:
        thesis_table[col] = aggregated_df[col]

    # Format the numbers into "Mean ± Std" strings
    for metric in target_metrics:
        mean_series = aggregated_df[metric]['mean']
        std_series = aggregated_df[metric]['std']
        
        # Round to 2 decimal places. If std is NaN (like for baselines where it might not vary), use 0.0
        thesis_table[f'{metric} (Mean ± Std)'] = (
            mean_series.round(2).astype(str) + " ± " + std_series.fillna(0).round(2).astype(str)
        )
        
        # Replace instances where the mean was 'nan' (from the 'N/A' strings) back to 'N/A'
        thesis_table[f'{metric} (Mean ± Std)'] = thesis_table[f'{metric} (Mean ± Std)'].replace('nan ± 0.0', 'N/A')

    # Save the formatted table to the root results folder
    output_path = os.path.join(RESULTS_DIR, "thesis_results_summary.csv")
    thesis_table.to_csv(output_path, index=False)
    
    print(f"\n✓ Summary table successfully generated and saved to {output_path}")

if __name__ == "__main__":
    aggregate_ablation_results()