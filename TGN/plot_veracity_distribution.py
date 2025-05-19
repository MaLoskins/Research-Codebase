#!/usr/bin/env python3
"""
Plot veracity (KDE distribution) of posts over time for PHEME events
using smoothed curves and an "All Posts" line.

This script reads the output of `preprocess_tgn_pheme.py` and generates
a grid of plots, one for each PHEME event, showing the smoothed density of
rumour, non-rumour, and all posts (interaction events) over time.
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde


def format_time_value(seconds: float, unit: str) -> float:
    """Converts seconds to the specified time unit."""
    if unit == "minutes":
        return seconds / 60
    if unit == "hours":
        return seconds / 3600
    if unit == "days":
        return seconds / (3600 * 24)
    return seconds


def plot_event_veracity_kde(
    ax: plt.Axes,
    event_name: str,
    event_dir: Path,
    time_unit_str: str,
    kde_points: int = 200, # Number of points to evaluate KDE
    bandwidth_method: str | float = 'scott', # KDE bandwidth
):
    """Plots veracity KDE for a single event on the given Axes object."""
    events_file = event_dir / "events.csv"
    event_labels_file = event_dir / "event_labels.npy"

    # --- 1. Load Data ---
    if not events_file.exists() or not event_labels_file.exists():
        ax.text(0.5, 0.5, "Data not found", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_title(event_name.capitalize(), fontsize=14, fontweight='medium')
        ax.set_xticks([]); ax.set_yticks([])
        return

    try:
        events_df = pd.read_csv(events_file)
        veracity_labels = np.load(event_labels_file)
    except Exception as e:
        ax.text(0.5, 0.5, "Error loading data", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_title(event_name.capitalize(), fontsize=14, fontweight='medium')
        ax.set_xticks([]); ax.set_yticks([])
        return

    if events_df.empty:
        ax.text(0.5, 0.5, "No event data", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_title(event_name.capitalize(), fontsize=14, fontweight='medium')
        ax.set_xticks([]); ax.set_yticks([])
        return

    if len(veracity_labels) != len(events_df):
        ax.text(0.5, 0.5, "Data length mismatch", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_title(event_name.capitalize(), fontsize=14, fontweight='medium')
        ax.set_xticks([]); ax.set_yticks([])
        return

    # --- 2. Prepare Data ---
    events_df["time_value"] = events_df["timestamp"].apply(lambda x: format_time_value(x, time_unit_str))
    events_df["veracity"] = veracity_labels
    events_df.dropna(subset=["time_value"], inplace=True)

    if events_df.empty:
        ax.text(0.5, 0.5, "No event data after time conversion", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_title(event_name.capitalize(), fontsize=14, fontweight='medium'); ax.set_xticks([]); ax.set_yticks([])
        return

    time_all_posts = events_df["time_value"].values
    rumour_times = events_df[events_df["veracity"] == 1]["time_value"].values
    non_rumour_times = events_df[events_df["veracity"] == 0]["time_value"].values

    # --- 3. Determine Plotting Range ---
    min_t = np.min(time_all_posts) if len(time_all_posts) > 0 else 0
    max_t = np.max(time_all_posts) if len(time_all_posts) > 0 else 1
    if min_t == max_t: # Handle case where all events are at one point
        # Extend range slightly for KDE visualization
        span = max(1.0, abs(min_t * 0.2)) # Add 20% span or 1 unit
        min_t -= span / 2
        max_t += span / 2
        if min_t < 0 and np.min(time_all_posts) >= 0: # ensure plot starts at 0 if original data did
             max_t = max_t - min_t # shift range to start at 0
             min_t = 0
             
    eval_points = np.linspace(min_t, max_t, kde_points)

    # --- 4. Calculate KDEs ---
    # Note: KDE gives a density. To make y-axis interpretable as "intensity" or proportional to count,
    # we can scale by the number of points in each category. This is optional.
    # For now, we plot the raw density. The y-axis will be "Density".

    kde_rumour_y = np.zeros_like(eval_points)
    if len(rumour_times) > 1: # KDE needs at least 2 points for variance
        try:
            kde_rumour = gaussian_kde(rumour_times, bw_method=bandwidth_method)
            kde_rumour_y = kde_rumour(eval_points)
        except Exception as e: # Catch potential LinAlgError if all points are identical
            print(f"Warning: KDE failed for rumours in {event_name} (may have too few unique points): {e}")
            # Fallback: simple histogram-like representation if KDE fails for few points
            if len(rumour_times) > 0:
                 hist_r, _ = np.histogram(rumour_times, bins=eval_points)
                 kde_rumour_y = np.interp(eval_points, (_[1:] + _[:-1])/2, hist_r/np.sum(hist_r) if np.sum(hist_r)>0 else hist_r)


    kde_non_rumour_y = np.zeros_like(eval_points)
    if len(non_rumour_times) > 1:
        try:
            kde_non_rumour = gaussian_kde(non_rumour_times, bw_method=bandwidth_method)
            kde_non_rumour_y = kde_non_rumour(eval_points)
        except Exception as e:
            print(f"Warning: KDE failed for non-rumours in {event_name}: {e}")
            if len(non_rumour_times) > 0:
                hist_nr, _ = np.histogram(non_rumour_times, bins=eval_points)
                kde_non_rumour_y = np.interp(eval_points, (_[1:] + _[:-1])/2, hist_nr/np.sum(hist_nr) if np.sum(hist_nr)>0 else hist_nr)


    kde_all_y = np.zeros_like(eval_points)
    if len(time_all_posts) > 1:
        try:
            kde_all = gaussian_kde(time_all_posts, bw_method=bandwidth_method)
            kde_all_y = kde_all(eval_points)
        except Exception as e:
            print(f"Warning: KDE failed for all posts in {event_name}: {e}")
            if len(time_all_posts) > 0:
                hist_a, _ = np.histogram(time_all_posts, bins=eval_points)
                kde_all_y = np.interp(eval_points, (_[1:] + _[:-1])/2, hist_a/np.sum(hist_a) if np.sum(hist_a)>0 else hist_a)
    
    # Normalize densities to prevent extreme y-axis scales if one category dominates heavily
    # This makes the shapes comparable, but loses direct proportionality to count.
    # If you want proportionality to count, multiply each kde_y by its respective len(times).
    # For fluid shapes, relative density is often more visually useful.
    # Let's scale them so the max of "All Posts" is around 1 for better y-axis consistency if desired
    # Or, plot raw densities and let y-axis adjust. For now, raw densities.

    # --- 5. Plotting ---
    color_rumour = "#e41a1c"
    color_non_rumour = "#377eb8"
    color_all = "#4daf4a" # Green

    # Non-Rumour
    ax.fill_between(eval_points, 0, kde_non_rumour_y, color=color_non_rumour, alpha=0.3, label="Non-Rumour Posts (density)")
    ax.plot(eval_points, kde_non_rumour_y, color=color_non_rumour, linewidth=1.5)

    # Rumour
    ax.fill_between(eval_points, 0, kde_rumour_y, color=color_rumour, alpha=0.3, label="Rumour Posts (density)")
    ax.plot(eval_points, kde_rumour_y, color=color_rumour, linewidth=1.5)

    # All Posts Line (plotted on top)
    ax.plot(eval_points, kde_all_y, label="All Posts (density)", color=color_all, linewidth=2.0, linestyle='-', zorder=3)

    ax.set_title(f"{event_name.capitalize()}", fontsize=14, fontweight='medium')
    ax.set_xlabel(f"Time ({time_unit_str.capitalize()}) from Event Start", fontsize=10)
    ax.set_ylabel("Estimated Density", fontsize=10) # Y-axis is density
    
    handles, labels = ax.get_legend_handles_labels()
    # Example reorder: order = [labels.index("All Posts (density)"), labels.index("Rumour Posts (density)"), labels.index("Non-Rumour Posts (density)")]
    # ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=8, loc='upper right')
    ax.legend(fontsize=8, loc='upper right')


    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.grid(True, linestyle="--", alpha=0.5, linewidth=0.7)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=min_t, right=max_t) # Ensure x-axis covers the evaluation range

    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune='both'))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, min_n_ticks=3, prune='upper'))


def main():
    parser = argparse.ArgumentParser(
        description="Plot veracity (KDE distribution) of posts over time."
    )
    parser.add_argument(
        "--input-dir", type=Path, default="data_tgn",
        help="Directory containing preprocessed TGN data subfolders.",
    )
    parser.add_argument(
        "--output-plot", type=Path, default="pheme_veracity_kde_curves.png",
        help="Path to save the output plot.",
    )
    parser.add_argument(
        "--time-unit", type=str, default="hours",
        choices=["seconds", "minutes", "hours", "days"],
        help="Time unit for the x-axis.",
    )
    parser.add_argument(
        "--kde-points", type=int, default=200,
        help="Number of points to evaluate KDE at for smoothness."
    )
    parser.add_argument(
        "--kde-bandwidth", type=str, default="scott", # Can be 'scott', 'silverman', or a float
        help="Bandwidth estimation method for KDE (e.g., 'scott', 'silverman', or a scalar value)."
    )
    parser.add_argument(
        "--plot-title", type=str, default="Smoothed Distribution of Posts Over Time (KDE)",
        help="Main title for the plot grid.",
    )
    args = parser.parse_args()

    try:
        args.kde_bandwidth = float(args.kde_bandwidth)
    except ValueError:
        pass # Keep as string if not a float

    sns.set_theme(style="whitegrid", palette="muted")

    event_dirs = sorted(
        [d for d in args.input_dir.iterdir() if d.is_dir() and d.name != "all"],
        key=lambda x: x.name,
    )

    if not event_dirs:
        print(f"No event subdirectories found in {args.input_dir} (excluding 'all').")
        return

    num_events = len(event_dirs)
    if num_events == 0: print("No events to plot."); return

    ncols = min(3, num_events)
    nrows = math.ceil(num_events / ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 6, nrows * 4.8), squeeze=False
    )
    axes = axes.flatten()

    print(f"Found {num_events} events to plot: {[d.name for d in event_dirs]}")
    print(f"Using KDE for smoothed distributions (bandwidth: {args.kde_bandwidth}).")

    for i, event_dir in enumerate(event_dirs):
        plot_event_veracity_kde(
            axes[i], event_dir.name, event_dir, args.time_unit, args.kde_points, args.kde_bandwidth
        )

    for j in range(num_events, nrows * ncols):
        fig.delaxes(axes[j])

    fig.suptitle(args.plot_title, fontsize=20, fontweight="bold")
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    
    args.output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_plot, dpi=300)
    print(f"\nMaster plot saved to {args.output_plot}")
    # plt.show()

if __name__ == "__main__":
    main()