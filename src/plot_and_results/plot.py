import os
import pandas as pd
from typing import Dict
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

def plot_fw_vs_afw(df_fw, df_afw, show=True):
    """
    Plot convergence comparison between FW and AFW methods.

    Parameters:
        df_fw (DataFrame): Log data from the FW method.
        df_afw (DataFrame):  Log data from the AFW method.
        show (bool):  Whether to display the plot immediately (default: True).

    Returns:
        fig (Figure):  The matplotlib figure containing the two subplots:
                      - f(x) over iterations
                      - Frank-Wolfe gap over iterations
    """
    
    # Create copies to avoid modifying the original DataFrames
    df_fw = df_fw.copy()
    df_afw = df_afw.copy()

    # Set the method column
    df_fw["method"] = "FW"
    df_afw["method"] = "AFW"

    # Combine both DataFrames
    df_all = pd.concat([df_fw, df_afw], ignore_index=True)

    # Clip small or zero gaps to avoid issues when plotting on a log scale
    df_all["gap"] = df_all["gap"].clip(lower=1e-16)
    df_fw["gap"] = df_fw["gap"].clip(lower=1e-16)
    df_afw["gap"] = df_afw["gap"].clip(lower=1e-16)

    # Subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Palette
    color_fw = "#fb2f38"   # red
    color_afw = "#2c52b3"  # blue

    # Plot objective function value over iterations
    axes[0].plot(df_fw['iteration'], df_fw['f(x)'], label='FW', color=color_fw, linewidth=1, alpha = 0.9)
    axes[0].plot(df_afw['iteration'], df_afw['f(x)'], label='AFW', color=color_afw, linewidth=1, linestyle='--', alpha = 0.9)
    axes[0].set_title('Objective Function Value')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('f(x)')
    axes[0].legend()

    # Plot F-W gap 
    axes[1].plot(df_fw['iteration'], df_fw['gap'], label='FW', color=color_fw, linewidth=1, alpha = 0.9)
    axes[1].plot(df_afw['iteration'], df_afw['gap'], label='AFW', color=color_afw, linewidth=1, linestyle='--', alpha = 0.9)
    
    if (df_all['gap'] > 0).any():
        axes[1].set_yscale('log')
        axes[1].set_title('Frank-Wolfe Gap (log scale)')
    else:
        axes[1].set_title('Frank-Wolfe Gap (not log-scaled — nonpositive values)')
        print("[Warning] All F-W gaps are ≤ 0. Log scale not applied.")

    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('FW gap')
    axes[1].legend()
    
    plt.tight_layout()
    
    if show:
        plt.show()

    return fig

def save_comparative_plots(results_by_config, output_dir):
    """
    Save comparison plots between FW and AFW for each configuration.

    Parameters:
        results_by_config (dict):  Dictionary mapping configuration keys to method results,
                                    e.g., {config_key: {'fw': df_fw, 'afw': df_afw}}.
        output_dir (str):  Directory in which to save the plots as PNG files.
    """
    
    for config_key, method_dfs in results_by_config.items():
        if "fw" in method_dfs and "afw" in method_dfs:
            fig = plot_fw_vs_afw(method_dfs["fw"], method_dfs["afw"], show=False)
            fig_path = os.path.join(output_dir, f"plot_{config_key}.png")
            fig.savefig(fig_path)
            plt.close(fig)

def plot_all_convergence_curves(results_by_config: Dict[str, Dict[str, pd.DataFrame]], show=True):
    """
    Plot all F-W gap convergence curves (log scale) in a single figure,
    comparing FW and AFW across multiple configurations.
    
    Parameters:
        results_by_config (dict):  Dictionary of the form:
                            {
                                config_key: {
                                    'fw': DataFrame,
                                    'afw': DataFrame
                                },
                                ...
                            }
        show (bool):  Whether to display the plot immediately (default: True).

    Returns:
        fig : matplotlib Figure containing the plot.
    """
    
    palette = {
        'random': "#ff0033",                 # red
        'internal': "#0066ff",               # blue
        'external_sum_simplex': "#000000",   # green
        'external_non_negative': "#ff9900",  # orange
        'external_all': "#00ff66"            # black
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    # Dictionary to store one representative line per configuration for legend
    config_handles = {}
    
    # Loop over configurations and plot both curves
    for config_key, method_dfs in results_by_config.items():
        color = palette.get(config_key, "#888888")

        # Plot FW with solid line (-)
        if "fw" in method_dfs:
            df_fw = method_dfs["fw"].copy()
            df_fw["gap"] = df_fw["gap"].clip(lower=1e-16)
            ax.plot(df_fw["iteration"], df_fw["gap"],
                    color=color, linestyle='-', alpha=0.8, linewidth=0.9)

        # Plot AFW with dashed line (--)
        if "afw" in method_dfs:
            df_afw = method_dfs["afw"].copy()
            df_afw["gap"] = df_afw["gap"].clip(lower=1e-16)
            ax.plot(df_afw["iteration"], df_afw["gap"],
                    color=color, linestyle='--', alpha=0.8, linewidth=0.9)

        # Save one line per config for the legend
        config_handles[config_key] = mlines.Line2D([], [], color=color, linestyle='-', label=config_key)
        
    fw_afw_key = '\u2014 FW, \u2013\u2013 AFW' # — FW, -- AFW
    all_handles = list(config_handles.values())
    
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Frank-Wolfe gap")
    
    # Legend creation
    ax.legend(handles=all_handles, loc="upper right", fontsize=8, title=f"q-gen strategy ({fw_afw_key})")

    plt.tight_layout()
    if show:
        plt.show()

    return fig
