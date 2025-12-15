import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from market_model import DataMarket, Participants
from mesa.batchrunner import batch_run
import json # To parse persona_dist string for a better legend
from typing import List, Dict, Any, Optional
from matplotlib.axes import Axes

# Set default font for English
plt.rcParams['font.family'] = 'sans-serif'

# Original functions (kept as is, but titles will be English)
def final_holders(results_df: pd.DataFrame, dist: str) -> pd.DataFrame:
    final_step = results_df['Step'].max()

    final_df = results_df[results_df['Step'] == final_step]
    final_df = final_df[final_df['persona_dist'] == str(dist)]
    final_df = final_df[final_df["AgentID"] == 1]

    mean_holders_by_price = final_df.groupby('initial_price')['Holders'].mean()
    mean_holders_by_price = mean_holders_by_price/1000
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    mean_holders_by_price.plot(
        kind='line',
        ax=ax,
        marker='o',
        title='Price vs. Mean Final Holders',
    )

    ax.set_xlabel('Initial Price')
    ax.set_ylabel('Mean Final Holders (in thousands)')
    ax.set_xscale('log')
    plt.grid(True)
    plt.show()

    print("Mean final holders by price:")
    return pd.DataFrame(mean_holders_by_price)
    
def provider_profit(results_df: pd.DataFrame, dist: str) -> pd.DataFrame:
    final_step = results_df['Step'].max()

    final_df = results_df[results_df['Step'] == final_step]
    final_df = final_df[final_df['persona_dist'] == str(dist)]
    final_df = final_df[final_df["AgentID"] == 1]
    
    mean_revenue_by_price = final_df.groupby('initial_price')['ProviderRevenue'].mean()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    mean_revenue_by_price.plot(
        kind='line',
        ax=ax,
        marker='o',
        color='green',
        title='Provider Profit vs Initial Price'
    )

    ax.set_xlabel('Initial Price', fontsize=12)
    ax.set_ylabel('Mean Provider Profit', fontsize=12)
    ax.set_xscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    print("Mean provider profit by price:")
    return pd.DataFrame(mean_revenue_by_price)

def market_participant_welfare(results_df: pd.DataFrame, dist: str) -> pd.DataFrame:
    final_step = results_df['Step'].max()

    final_df = results_df[results_df['Step'] == final_step]
    final_df = final_df[final_df['persona_dist'] == str(dist)]
    
    iteration_average = (
        final_df.groupby(['initial_price', 'AgentID'])["CurrentUtility"]
            .mean()
            .rename("AvgCurrentUtility")
            .reset_index()
    )    
    welfare_avg = iteration_average.groupby('initial_price')["AvgCurrentUtility"].mean()
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7)) 
    welfare_avg.plot(
        kind='line',
        ax=ax,
        marker='o',
        color='orange',
        title='Market Participant Welfare vs Initial Price'
    )
    ax.set_xlabel('Initial Price')
    ax.set_ylabel('Mean Market Participant Welfare')
    ax.set_xscale('log')
    plt.grid(True)
    plt.show()
    
    return pd.DataFrame(welfare_avg)

# --- New functions for plotting all scenarios on one graph ---

# Helper function to create a readable label for persona_dist
def get_persona_label(persona_dict_str: str) -> str:
    try:
        if persona_dict_str.startswith('[') and persona_dict_str.endswith(']'):
            persona_dict_str = persona_dict_str[1:-1]
        
        persona_dict = json.loads(persona_dict_str.replace("'", '"'))
        
        labels = []
        if persona_dict.get('Bandwagon', 0) > 0.5:
            labels.append("Bandwagon Dominant")
        elif persona_dict.get('Snob', 0) > 0.5:
            labels.append("Snob Dominant")
        elif persona_dict.get('Neutral', 0) > 0.5:
            labels.append("Neutral Dominant")
        
        if not labels:
            parts = []
            if 'Bandwagon' in persona_dict: parts.append(f"B:{persona_dict['Bandwagon']:.1f}")
            if 'Neutral' in persona_dict: parts.append(f"N:{persona_dict['Neutral']:.1f}")
            if 'Snob' in persona_dict: parts.append(f"S:{persona_dict['Snob']:.1f}")
            return ", ".join(parts) if parts else persona_dict_str
        return ", ".join(labels)
    except json.JSONDecodeError:
        return persona_dict_str

def plot_metric_by_persona(results_df: pd.DataFrame, metric_col: str, title: str, ylabel: str, is_ratio: bool = False) -> None:
    results_df['persona_label'] = results_df['persona_dist'].apply(get_persona_label)

    final_step = results_df['Step'].max()
    final_df = results_df[results_df['Step'] == final_step].copy()

    if metric_col == 'Holders' and is_ratio:
        final_df['HoldersRatio'] = final_df['Holders'] / final_df['num_agents']
        metric_col_to_plot = 'HoldersRatio'
    else:
        metric_col_to_plot = metric_col

    if metric_col == 'CurrentUtility':
        agent_avg_utility_per_run = final_df.groupby(['RunId', 'initial_price', 'persona_label'])[metric_col_to_plot].mean().reset_index()
        plot_data = agent_avg_utility_per_run.groupby(['initial_price', 'persona_label'])[metric_col_to_plot].mean().reset_index()
    else:
        model_level_data = final_df.drop_duplicates(subset=['RunId', 'initial_price', 'persona_label'])
        plot_data = model_level_data.groupby(['initial_price', 'persona_label'])[metric_col_to_plot].mean().reset_index()

    # Poster styling
    plt.rcParams.update({'font.size': 24}) # Base font size
    plt.figure(figsize=(14, 10), dpi=300)
    
    # Create the plot
    ax = sns.lineplot(
        data=plot_data,
        x='initial_price',
        y=metric_col_to_plot,
        hue='persona_label',
        marker='o',
        palette='bright', # High contrast palette
        linewidth=5,
        markersize=12
    )

    # plt.title(title, fontsize=32, fontweight='bold', pad=20) # Title removed as per user request
    plt.xlabel('Initial Price (Log Scale)', fontsize=32, fontweight='bold', labelpad=15)
    plt.ylabel(ylabel, fontsize=32, fontweight='bold', labelpad=15)
    plt.xscale('log')
    
    # Grid and Ticks
    plt.grid(True, which='both', linestyle='--', linewidth=1.5)
    ax.tick_params(axis='both', which='major', labelsize=28, width=2, length=6)
    ax.tick_params(axis='both', which='minor', width=1, length=4)
    
    # Legend styling
    plt.legend(title='Persona Scenario', title_fontsize=28, fontsize=24)
    
    plt.tight_layout()
    plt.show()

# --- Plotting functions with English labels ---

def plot_final_holders_by_persona(results_df: pd.DataFrame) -> None:
    plot_metric_by_persona(results_df, 'Holders', 'Final Holder Ratio vs. Price', 'Final Holder Ratio', is_ratio=True)

def plot_provider_profit_by_persona(results_df: pd.DataFrame) -> None:
    plot_metric_by_persona(results_df, 'ProviderRevenue', 'Provider Profit vs. Price', 'Provider Profit')

def plot_market_welfare_by_persona(results_df: pd.DataFrame) -> None:
    plot_metric_by_persona(results_df, 'CurrentUtility', 'Market Participant Welfare vs. Price', 'Market Welfare')

# --- New function for 3x3 grid plot (Transposed and in English) ---

def _plot_holders_on_ax(ax: Axes, df: pd.DataFrame, num_agents: int, color: str) -> None:
    """Helper to plot final holders on a given axes."""
    df = df.copy()
    model_level_data = df.drop_duplicates(subset=['RunId', 'initial_price'])
    model_level_data['HoldersRatio'] = model_level_data['Holders'] / num_agents
    plot_data = model_level_data.groupby('initial_price')['HoldersRatio'].mean()
    plot_data.plot(kind='line', ax=ax, marker='o', color=color, linewidth=5, markersize=12)

def _plot_profit_on_ax(ax: Axes, df: pd.DataFrame, color: str) -> None:
    """Helper to plot provider profit on a given axes."""
    model_level_data = df.drop_duplicates(subset=['RunId', 'initial_price'])
    plot_data = model_level_data.groupby('initial_price')['ProviderRevenue'].mean()
    plot_data.plot(kind='line', ax=ax, marker='o', color=color, linewidth=5, markersize=12)

def _plot_welfare_on_ax(ax: Axes, df: pd.DataFrame, color: str) -> None:
    """Helper to plot market welfare on a given axes."""
    agent_avg_utility_per_run = df.groupby(['RunId', 'initial_price'])['CurrentUtility'].mean().reset_index()
    plot_data = agent_avg_utility_per_run.groupby('initial_price')['CurrentUtility'].mean()
    plot_data.plot(kind='line', ax=ax, marker='o', color=color, linewidth=5, markersize=12)

def plot_scenarios_grid(results_df: pd.DataFrame, persona_dists: List[Any]) -> None:
    """
    Plots a 3x3 grid of graphs: 3 metrics (rows) x 3 scenarios (columns).
    Optimized for Poster (High DPI, Large Fonts, Bold Lines).
    """
    # Set global font size for this plot context
    plt.rcParams.update({'font.size': 28})
    
    persona_dist_strs = [str(d) for d in persona_dists]

    # Increase DPI for poster quality
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(24, 20), sharex=True, sharey='row', dpi=300)
    
    # Title removed as per user request (will be added via text box)
    # fig.suptitle('Comparison of 3 Scenarios and 3 Metrics', fontsize=24, y=0.97)

    final_step = results_df['Step'].max()
    final_df = results_df[results_df['Step'] == final_step].copy()
    
    num_agents = final_df['num_agents'].iloc[0] if not final_df.empty else 1000

    # Use high-contrast, distinct colors
    metrics = [
        {'name': 'Holders', 'title': 'Final Holder Ratio', 'plot_func': _plot_holders_on_ax, 'color': 'navy'},
        {'name': 'ProviderRevenue', 'title': 'Provider Profit', 'plot_func': _plot_profit_on_ax, 'color': '#D62728'}, # Bold Red
        {'name': 'CurrentUtility', 'title': 'Social Welfare', 'plot_func': _plot_welfare_on_ax, 'color': 'darkgreen'}
    ]
    
    scenario_titles = [get_persona_label(s) for s in persona_dist_strs]

    for i, metric in enumerate(metrics): # Rows are metrics
        for j, persona_dist_str in enumerate(persona_dist_strs): # Columns are scenarios
            ax = axes[i, j]
            scenario_df = final_df[final_df['persona_dist'] == persona_dist_str]
            
            # Larger Font Sizes for Poster
            if i == 0:
                ax.set_title(scenario_titles[j], fontsize=40, fontweight='bold', pad=20)
            
            if j == 0:
                ax.set_ylabel(metric['title'], fontsize=36, fontweight='bold', labelpad=15)
            else:
                ax.set_ylabel('')

            if scenario_df.empty:
                print(f"Warning: No data found for scenario: {persona_dist_str}")
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=30)
            else:
                if metric['name'] == 'Holders':
                    metric['plot_func'](ax, scenario_df, num_agents, color=metric['color'])
                else:
                    metric['plot_func'](ax, scenario_df, color=metric['color'])

            ax.grid(True, which='both', linestyle='--', linewidth=1.5)
            ax.set_xscale('log')
            
            # Increase tick label size
            ax.tick_params(axis='both', which='major', labelsize=32, width=2, length=6)
            ax.tick_params(axis='both', which='minor', width=1, length=4)

            if i < 2:
                ax.set_xlabel('')
            else:
                ax.set_xlabel('Initial Price (Log Scale)', fontsize=36, fontweight='bold', labelpad=15)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()