import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os # For checking file existence



# --- Configuration ---
STUDY_NAME = "FullDySAT_Global_AllEvents"  # This should match your Optuna study name
DB_FILENAME = f"{STUDY_NAME.replace(' ', '_')}.db"
OUTPUT_DIR = "optuna_plots_FullDySAT_Global" # Give it a specific output dir
# ... rest of the script

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_optuna_plots(study_name: str, db_filename: str, output_dir: str):
    """
    Loads an Optuna study and generates various visualizations.
    """
    storage_name = f"sqlite:///{db_filename}"
    
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        print(f"Successfully loaded study '{study_name}' with {len(study.trials)} trials.")
    except Exception as e:
        print(f"Error loading study '{study_name}' from '{db_filename}': {e}")
        print("Please ensure the study name and database file are correct and the study has run.")
        return

    if not any(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials):
        print(f"Study '{study_name}' has no completed trials. Cannot generate plots.")
        return

    # --- General Matplotlib/Seaborn Styling ---
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    plt.rcParams['figure.figsize'] = (12, 7) # Slightly larger default
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.family'] = 'serif' 
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12


    plot_prefix = os.path.join(output_dir, f"{study_name}_")

    # 1. Optimization History Plot
    try:
        fig_history = optuna.visualization.plot_optimization_history(study)
        fig_history.update_layout(title_font_size=20, font_size=14)
        fig_history.write_image(f"{plot_prefix}optimization_history.png", scale=2, width=1000, height=600)
        print(f"Saved {plot_prefix}optimization_history.png")
    except Exception as e:
        print(f"Could not generate optimization history plot: {e}")

    # 2. Parallel Coordinate Plot
    try:
        # Consider only completed trials with valid values for parallel coordinate
        completed_trials_df = study.trials_dataframe()
        completed_trials_df = completed_trials_df[completed_trials_df['state'] == 'COMPLETE'].dropna(subset=['value'])

        if not completed_trials_df.empty:
            # Create a temporary study object with only these trials if direct filtering in plot function is problematic
            # This is a workaround if the plot function doesn't handle NaNs or non-COMPLETE trials well.
            # For now, let's assume the plot function handles it or we pass filtered trials.
            
            # Get params that were actually tuned
            param_names_to_plot = list(study.best_params.keys())

            fig_parallel = optuna.visualization.plot_parallel_coordinate(study, params=param_names_to_plot)
            fig_parallel.update_layout(title_font_size=20, font_size=14)
            fig_parallel.write_image(f"{plot_prefix}parallel_coordinate.png", scale=2, width=1200, height=700)
            print(f"Saved {plot_prefix}parallel_coordinate.png")
        else:
            print("No completed trials with valid values to generate parallel coordinate plot.")
    except Exception as e:
        print(f"Could not generate parallel coordinate plot: {e}")

    # Get hyperparameter importances to decide which slices/contours are most interesting
    try:
        param_importances_dict = optuna.importance.get_param_importances(study)
        # Sort by importance (descending)
        important_params_sorted = [p for p, _ in sorted(param_importances_dict.items(), key=lambda item: item[1], reverse=True)]
    except Exception as e:
        print(f"Could not calculate parameter importances: {e}. Skipping some plots.")
        important_params_sorted = []


    # 3. Slice Plot
    if important_params_sorted:
        # Plot slices for top N important params
        slice_params_to_plot = important_params_sorted[:min(5, len(important_params_sorted))] 
        try:
            if slice_params_to_plot:
                fig_slice = optuna.visualization.plot_slice(study, params=slice_params_to_plot)
                fig_slice.update_layout(title_font_size=20, font_size=14)
                fig_slice.write_image(f"{plot_prefix}slice_plot_top_params.png", scale=2, width=1200, height=700 if len(slice_params_to_plot)>2 else 500)
                print(f"Saved {plot_prefix}slice_plot_top_params.png for params: {slice_params_to_plot}")

                # Individual Matplotlib Slice Plots for more control (e.g., for the top 2)
                for param_name in slice_params_to_plot[:min(2, len(slice_params_to_plot))]:
                    plt.figure()
                    ax = optuna.visualization.matplotlib.plot_slice(study, params=[param_name])
                    ax.set_title(f"Slice Plot: {param_name}", fontsize=plt.rcParams['axes.titlesize'])
                    # ax.set_xlabel(param_name, fontsize=plt.rcParams['axes.labelsize']) # Optuna might set this
                    # ax.set_ylabel("Objective Value", fontsize=plt.rcParams['axes.labelsize'])
                    plt.tight_layout()
                    plt.savefig(f"{plot_prefix}slice_{param_name}_mpl.png")
                    plt.close()
                    print(f"Saved {plot_prefix}slice_{param_name}_mpl.png")
            else:
                print("No parameters selected for slice plot.")
        except Exception as e:
            print(f"Could not generate slice plot: {e}")
    else:
        print("No parameter importances available to select parameters for slice plot.")


    # 4. Hyperparameter Importance Plot
    if important_params_sorted:
        try:
            plt.figure()
            ax = optuna.visualization.matplotlib.plot_param_importances(study)
            ax.set_title("Hyperparameter Importances")
            plt.tight_layout()
            plt.savefig(f"{plot_prefix}param_importances_mpl.png")
            plt.close()
            print(f"Saved {plot_prefix}param_importances_mpl.png")
        except Exception as e:
            print(f"Could not generate param importances plot: {e}")

    # 5. Contour Plot (for pairs of hyperparameters)
    if len(important_params_sorted) >= 2:
        # Plot contour for the top 2 most important parameters
        param1, param2 = important_params_sorted[0], important_params_sorted[1]
        try:
            plt.figure()
            ax = optuna.visualization.matplotlib.plot_contour(study, params=[param1, param2])
            ax.set_title(f"Contour Plot: {param1} vs {param2}")
            # You can further customize titles, labels, and colorbar if needed
            # e.g. cbar = ax.collections[0].colorbar; cbar.set_label('Avg. Test Accuracy')
            plt.tight_layout()
            plt.savefig(f"{plot_prefix}contour_{param1}_vs_{param2}_mpl.png")
            plt.close()
            print(f"Saved {plot_prefix}contour_{param1}_vs_{param2}_mpl.png")
        except Exception as e:
            print(f"Could not generate contour plot for {param1} vs {param2}: {e}")
    elif important_params_sorted:
        print("Not enough important parameters for a contour plot (need at least 2).")

    print(f"\nAll plotting attempts finished. Plots saved in '{output_dir}' directory.")

if __name__ == "__main__":
    # Check if kaleido is installed, as it's needed for fig.write_image()
    try:
        import kaleido
    except ImportError:
        print("WARNING: kaleido is not installed. Static image export for Plotly figures will fail.")
        print("Please install it: pip install kaleido")
        # You might choose to exit or only generate matplotlib plots if kaleido is missing.

    generate_optuna_plots(STUDY_NAME, DB_FILENAME, OUTPUT_DIR)

    # Additionally, save the trials dataframe to CSV
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=f"sqlite:///{DB_FILENAME}")
        df_trials = study.trials_dataframe()
        df_trials_path = os.path.join(OUTPUT_DIR, f"{STUDY_NAME}_trials_dataframe.csv")
        df_trials.to_csv(df_trials_path, index=False)
        print(f"Trials dataframe saved to {df_trials_path}")
    except Exception as e:
        print(f"Could not save trials dataframe: {e}")