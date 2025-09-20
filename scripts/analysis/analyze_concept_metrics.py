#!/usr/bin/env python3
"""
Analyze concept metrics results and create comprehensive tables, plots, and aggregated analysis.

This script collects all concept_metrics.yaml files and creates:
1. Summary tables of effectiveness and entanglement by method/concept/model
2. Pareto frontier plots of effectiveness vs entanglement  
3. Method comparison tables and visualizations
4. Statistical analysis across experiments

Usage:
    python analyze_concept_metrics.py [--experiments-dir PATH] [--output-prefix PREFIX]
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def collect_concept_metrics_files(search_dirs: List[str] = None) -> List[str]:
    """Collect all concept_metrics.yaml files from specified directories."""
    if search_dirs is None:
        search_dirs = ["./experiments"]
    
    metrics_files = []
    
    for base_dir in search_dirs:
        if os.path.isfile(base_dir) and base_dir.endswith('concept_metrics.yaml'):
            # Direct file path
            metrics_files.append(base_dir)
        elif os.path.isdir(base_dir):
            # Directory - search recursively
            for root, dirs, files in os.walk(base_dir):
                if 'concept_metrics.yaml' in files:
                    # Skip baseline directories
                    if 'baseline_' not in root:
                        metrics_files.append(os.path.join(root, 'concept_metrics.yaml'))
        else:
            # Check if it's an experiment directory
            metrics_file = os.path.join(base_dir, 'concept_metrics.yaml')
            if os.path.exists(metrics_file):
                metrics_files.append(metrics_file)
    
    print(f"Found {len(metrics_files)} concept metrics files")
    return metrics_files


def parse_experiment_name(exp_path: str) -> Tuple[str, str, str]:
    """Parse experiment name to extract method, concept, and model.

    Assumes exp_path is the full path to concept_metrics.yaml file.

    """
    # Get the parent directory (timestamp dir), then its parent (experiment name)
    timestamp_dir = os.path.dirname(exp_path)
    exp_name = os.path.basename(os.path.dirname(timestamp_dir))
    
    # Handle different naming patterns: method_concept_model or method_variant_concept_model
    parts = exp_name.split('_')
    
    if len(parts) >= 3:
        # Extract model (last part)
        model = parts[-1]
        
        # Known method variants that should be treated as complete method names
        known_methods = {
            'ace_cosmic', 'caa_conditional', 'caa_cosmic', 'dim_conditional', 
            'dim_cosmic', 'dim_nokl', 'lat_conditional', 'pca_cosmic', 'pca_new_cosmic',
            'lat_nokl', 'ace_nokl', 'caa_nokl', 'pca_nokl',
            'pca_conditional', 'ace_conditional'
        }
        
        # Try to match known method variants first (2-part methods)
        method = parts[0]
        concept_start_idx = 1
        
        if len(parts) >= 4:  # At least method_variant_concept_model
            two_part_method = f"{parts[0]}_{parts[1]}"
            if two_part_method in known_methods:
                method = two_part_method
                concept_start_idx = 2
        
        # Extract concept (everything between method and model)
        concept = '_'.join(parts[concept_start_idx:-1])
        
        return method, concept, model
    else:
        raise ValueError(f"Cannot parse experiment name: {exp_name}")


def load_metrics_data(metrics_files: List[str]) -> Dict:
    """Load all metrics data organized by model, concept, and method."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for file_path in metrics_files:
        try:
            with open(file_path, 'r') as f:
                metrics = yaml.safe_load(f)
            
            # Parse experiment info from directory path
            method, concept, model = parse_experiment_name(file_path)
            
            # Store the metrics data
            data[model][concept][method] = {
                'effectiveness': metrics.get('effectiveness', 0.0),
                'entanglement_all': metrics.get('entanglement_all', 0.0),
                'entanglement_negative_only': metrics.get('entanglement_negative_only', 0.0),
                'entanglement_positive_only': metrics.get('entanglement_positive_only', 0.0),
                'file_path': file_path
            }
            
            print(f"Loaded: {model} - {concept} - {method}")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return dict(data)


def create_summary_table(data: Dict) -> pd.DataFrame:
    """Create a comprehensive summary table of all metrics."""
    summary_data = []
    
    for model in data:
        for concept in data[model]:
            for method in data[model][concept]:
                metrics = data[model][concept][method]
                summary_data.append({
                    'Model': model.replace('-', ' ').upper(),
                    'Concept': concept.replace('_', ' ').title(),
                    'Method': method.upper(),
                    'Effectiveness': metrics['effectiveness'] * 100,  # Convert to percentage
                    'Entanglement (All)': metrics['entanglement_all'] * 100,
                    'Entanglement (Negative)': metrics['entanglement_negative_only'] * 100,
                    'Entanglement (Positive)': metrics['entanglement_positive_only'] * 100
                })
    
    df = pd.DataFrame(summary_data)
    return df.sort_values(['Model', 'Concept', 'Method'])


def create_effectiveness_table(data: Dict) -> pd.DataFrame:
    """Create a pivot table showing effectiveness by model, concept, and method."""
    summary_df = create_summary_table(data)
    
    if summary_df.empty:
        return pd.DataFrame()
    
    # Pivot to get methods as columns
    pivot_table = summary_df.pivot_table(
        index=['Model', 'Concept'], 
        columns='Method', 
        values='Effectiveness',
        aggfunc='first'
    )
    
    return pivot_table


def create_entanglement_table(data: Dict, entanglement_type: str = 'entanglement_all') -> pd.DataFrame:
    """Create a pivot table showing entanglement by model, concept, and method."""
    summary_df = create_summary_table(data)
    
    if summary_df.empty:
        return pd.DataFrame()
    
    # Map entanglement type to column name
    col_map = {
        'entanglement_all': 'Entanglement (All)',
        'entanglement_negative_only': 'Entanglement (Negative)',
        'entanglement_positive_only': 'Entanglement (Positive)'
    }
    
    column_name = col_map.get(entanglement_type, 'Entanglement (All)')
    
    # Pivot to get methods as columns
    pivot_table = summary_df.pivot_table(
        index=['Model', 'Concept'], 
        columns='Method', 
        values=column_name,
        aggfunc='first'
    )
    
    return pivot_table


def create_method_comparison_table(data: Dict) -> pd.DataFrame:
    """Create a table comparing average performance across methods."""
    summary_df = create_summary_table(data)
    
    if summary_df.empty:
        return pd.DataFrame()
    
    # Group by method and calculate statistics
    method_stats = summary_df.groupby('Method').agg({
        'Effectiveness': ['mean', 'std', 'count'],
        'Entanglement (All)': ['mean', 'std'],
        'Entanglement (Negative)': ['mean', 'std'],
        'Entanglement (Positive)': ['mean', 'std']
    }).round(2)
    
    # Flatten column names
    method_stats.columns = ['_'.join(col).strip() for col in method_stats.columns]
    
    return method_stats


def create_visualization_plots(data: Dict, output_dir: str = "."):
    """Create comprehensive visualization plots."""
    summary_df = create_summary_table(data)
    
    if summary_df.empty:
        print("No data available for visualization")
        return
    
    # 1. Effectiveness vs Entanglement scatter plot
    plt.figure(figsize=(12, 8))
    
    # Color by method
    methods = summary_df['Method'].unique()
    colors = sns.color_palette("husl", len(methods))
    method_colors = dict(zip(methods, colors))
    
    for method in methods:
        method_data = summary_df[summary_df['Method'] == method]
        plt.scatter(method_data['Entanglement (All)'], method_data['Effectiveness'], 
                   label=method, alpha=0.7, s=100, c=[method_colors[method]])
    
    plt.xlabel('Entanglement (All Deviations) %')
    plt.ylabel('Effectiveness %')
    plt.title('Effectiveness vs Entanglement by Method')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Create subdirectories for other formats
    svg_dir = os.path.join(output_dir, 'svgs')
    pdf_dir = os.path.join(output_dir, 'pdfs')
    os.makedirs(svg_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    # Save in all formats
    plt.savefig(os.path.join(output_dir, 'effectiveness_vs_entanglement.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(svg_dir, 'effectiveness_vs_entanglement.svg'), bbox_inches='tight')
    plt.savefig(os.path.join(pdf_dir, 'effectiveness_vs_entanglement.pdf'), bbox_inches='tight')
    plt.close()
    
    # 2. Method comparison boxplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Effectiveness boxplot
    summary_df.boxplot(column='Effectiveness', by='Method', ax=axes[0,0])
    axes[0,0].set_title('Effectiveness by Method')
    axes[0,0].set_ylabel('Effectiveness %')
    
    # Entanglement boxplots
    summary_df.boxplot(column='Entanglement (All)', by='Method', ax=axes[0,1])
    axes[0,1].set_title('Entanglement (All) by Method')
    axes[0,1].set_ylabel('Entanglement %')
    
    summary_df.boxplot(column='Entanglement (Negative)', by='Method', ax=axes[1,0])
    axes[1,0].set_title('Entanglement (Negative) by Method')
    axes[1,0].set_ylabel('Entanglement %')
    
    summary_df.boxplot(column='Entanglement (Positive)', by='Method', ax=axes[1,1])
    axes[1,1].set_title('Entanglement (Positive) by Method')
    axes[1,1].set_ylabel('Entanglement %')
    
    plt.suptitle('Method Performance Comparison', y=1.02)
    plt.tight_layout()

    # Save in all formats
    plt.savefig(os.path.join(output_dir, 'method_comparison_boxplots.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(svg_dir, 'method_comparison_boxplots.svg'), bbox_inches='tight')
    plt.savefig(os.path.join(pdf_dir, 'method_comparison_boxplots.pdf'), bbox_inches='tight')
    plt.close()
    
    # 3. Heatmap of effectiveness by model and concept
    if len(summary_df['Model'].unique()) > 1 and len(summary_df['Concept'].unique()) > 1:
        eff_pivot = summary_df.pivot_table(values='Effectiveness', index='Concept', columns='Model', aggfunc='mean')
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(eff_pivot, annot=True, fmt='.1f', cmap='viridis')
        plt.title('Average Effectiveness by Model and Concept')
        plt.tight_layout()

        # Save in all formats
        plt.savefig(os.path.join(output_dir, 'effectiveness_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(svg_dir, 'effectiveness_heatmap.svg'), bbox_inches='tight')
        plt.savefig(os.path.join(pdf_dir, 'effectiveness_heatmap.pdf'), bbox_inches='tight')
        plt.close()
    
    print(f"Visualization plots saved to {output_dir}/")


def save_tables_to_files(data: Dict, output_prefix: str = "concept_metrics"):
    """Save all tables to various formats."""
    
    # Create summary table
    summary_df = create_summary_table(data)
    
    # Save to CSV
    summary_df.to_csv(f"{output_prefix}_summary.csv", index=False)
    
    # Create and save pivot tables
    effectiveness_table = create_effectiveness_table(data)
    entanglement_all_table = create_entanglement_table(data, 'entanglement_all')
    entanglement_neg_table = create_entanglement_table(data, 'entanglement_negative_only')
    entanglement_pos_table = create_entanglement_table(data, 'entanglement_positive_only')
    method_comparison = create_method_comparison_table(data)
    
    # Save pivot tables to CSV
    if not effectiveness_table.empty:
        effectiveness_table.to_csv(f"{output_prefix}_effectiveness.csv")
    if not entanglement_all_table.empty:
        entanglement_all_table.to_csv(f"{output_prefix}_entanglement_all.csv")
    if not entanglement_neg_table.empty:
        entanglement_neg_table.to_csv(f"{output_prefix}_entanglement_negative.csv")
    if not entanglement_pos_table.empty:
        entanglement_pos_table.to_csv(f"{output_prefix}_entanglement_positive.csv")
    if not method_comparison.empty:
        method_comparison.to_csv(f"{output_prefix}_method_comparison.csv")
    
    # Save to Markdown
    with open(f"{output_prefix}_analysis.md", 'w') as f:
        f.write("# Concept Metrics Analysis Results\n\n")
        
        f.write("## Summary Statistics\n\n")
        if not summary_df.empty:
            f.write(f"Total experiments: {len(summary_df)}\n")
            f.write(f"Models: {', '.join(summary_df['Model'].unique())}\n")
            f.write(f"Concepts: {', '.join(summary_df['Concept'].unique())}\n")
            f.write(f"Methods: {', '.join(summary_df['Method'].unique())}\n\n")
        
        f.write("## Method Comparison\n\n")
        if not method_comparison.empty:
            f.write(method_comparison.to_markdown(tablefmt="github"))
            f.write("\n\n")
        
        f.write("## Effectiveness by Model and Concept\n\n")
        if not effectiveness_table.empty:
            f.write(effectiveness_table.round(1).to_markdown(tablefmt="github"))
            f.write("\n\n")
        
        f.write("## Entanglement (All) by Model and Concept\n\n")
        if not entanglement_all_table.empty:
            f.write(entanglement_all_table.round(1).to_markdown(tablefmt="github"))
            f.write("\n\n")
    
    print(f"Tables saved with prefix: {output_prefix}")


def plot_pareto_frontier_with_variants(data: Dict, output_dir: str = "."):
    """Create Pareto frontier plots with method variants (normal, no_kl, conditional)."""
    summary_df = create_summary_table(data)
    
    if summary_df.empty:
        print("No data available for Pareto frontier plots")
        return
    
    # Define base methods and their variants
    base_methods = ['DIM', 'ACE', 'CAA', 'PCA', 'LAT']
    
    # Check if we have variant methods
    all_methods = summary_df['Method'].unique()
    has_nokl = any('_NOKL' in m for m in all_methods)
    has_conditional = any('_CONDITIONAL' in m for m in all_methods)
    
    if has_nokl or has_conditional:
        # Use variant plotting
        plot_pareto_with_method_variants(summary_df, output_dir)
    else:
        # Use original simple plotting
        plot_pareto_frontier(data, output_dir)


def plot_pareto_with_method_variants(summary_df: pd.DataFrame, output_dir: str = "."):
    """Create Pareto plots with shapes for methods and colors for variations."""

    # Define visual mappings - shapes for methods, colors for variants
    method_shapes = {
        'DIM': 'o',      # circle
        'ACE': 's',      # square
        'CAA': '^',      # triangle
        'PCA': 'D',      # diamond
        'LAT': 'v'       # triangle down
    }

    # Colorblind-friendly palette for variants (using Paul Tol's palette)
    variant_colors = {
        'normal': '#0173B2',      # blue
        'nokl': '#DE8F05',        # orange
        'conditional': '#029E73'   # green
    }

    variant_labels = {
        'normal': 'Standard',
        'nokl': 'No KL',
        'conditional': 'Conditional'
    }

    # Create plots for each model
    models = summary_df['Model'].unique()

    for model in models:
        model_data = summary_df[summary_df['Model'] == model]

        if len(model_data) < 2:
            continue

        fig, ax = plt.subplots(figsize=(12, 8))

        # AGGREGATE: Calculate average effectiveness and entanglement for each method
        method_averages = model_data.groupby('Method').agg({
            'Effectiveness': 'mean',
            'Entanglement (All)': 'mean'
        }).reset_index()

        # Process each aggregated method and its variants
        all_points = []
        plotted_points = {}  # Store locations to avoid label overlap

        for _, row in method_averages.iterrows():
            method_name = row['Method']

            # Determine base method and variant
            if '_NOKL' in method_name:
                base_method = method_name.replace('_NOKL', '')
                variant = 'nokl'
            elif '_CONDITIONAL' in method_name:
                base_method = method_name.replace('_CONDITIONAL', '')
                variant = 'conditional'
            elif '_COSMIC' in method_name:
                # skip
                continue
            else:
                base_method = method_name
                variant = 'normal'

            # Get shape and color (shape from method, color from variant)
            shape = method_shapes.get(base_method, 'o')
            color = variant_colors.get(variant, 'gray')

            # Plot point with larger size
            x, y = row['Entanglement (All)'], row['Effectiveness']
            ax.scatter(x, y,
                      marker=shape, c=color, s=250, alpha=0.9,
                      edgecolors='white', linewidth=1.5, zorder=5)

            # Smart labeling: only label points that are Pareto-optimal or interesting
            # Store point for later labeling decision
            plotted_points[method_name] = {'x': x, 'y': y, 'base': base_method, 'variant': variant}

            all_points.append({
                'variant': variant,
                'x': x,
                'y': y,
                'method': base_method,
                'label': method_name
            })

        # Find Pareto-optimal points for labeling
        pareto_optimal = set()
        for name, point in plotted_points.items():
            is_pareto = True
            for other_name, other_point in plotted_points.items():
                if name != other_name:
                    # Check if other point dominates this one
                    if other_point['x'] <= point['x'] and other_point['y'] >= point['y']:
                        if other_point['x'] < point['x'] or other_point['y'] > point['y']:
                            is_pareto = False
                            break
            if is_pareto:
                pareto_optimal.add(name)

        # Remove connecting lines - color grouping is sufficient for method identification

        # Draw single combined Pareto frontier across all variants
        all_points_for_frontier = [(p['x'], p['y'], p['label']) for p in all_points]

        if len(all_points_for_frontier) > 1:
            # Sort by x-coordinate (entanglement)
            all_points_for_frontier.sort(key=lambda p: p[0])

            # Find combined Pareto frontier points
            frontier_points = []
            max_eff = -float('inf')

            for x, y, label in all_points_for_frontier:
                if y > max_eff:
                    frontier_points.append((x, y))
                    max_eff = y

            # Draw single frontier line
            if len(frontier_points) > 1:
                fx, fy = zip(*frontier_points)
                ax.plot(fx, fy,
                       linestyle='-',
                       color='#333333',  # dark gray
                       alpha=0.8,
                       linewidth=3.0,
                       zorder=2,
                       label='Pareto Frontier')

        # Detect if we have multiple variants
        unique_variants = set(point['variant'] for point in plotted_points.values())
        has_multiple_variants = len(unique_variants) > 1

        # Create combined legend with cleaner organization
        legend_elements = []

        if has_multiple_variants:
            # Add header for methods (now shapes)
            legend_elements.append(plt.Line2D([0], [0], color='none', label='Methods:'))
            for method in ['DIM', 'ACE', 'CAA', 'PCA', 'LAT']:
                if method in method_shapes:
                    legend_elements.append(
                        plt.Line2D([0], [0], marker=method_shapes[method], color='w',
                                  markerfacecolor='#666666', markersize=10,
                                  label=f'  {method}', markeredgecolor='white', markeredgewidth=1.0)
                    )

            # Add spacer
            legend_elements.append(plt.Line2D([0], [0], color='none', label=' '))

            # Add header for variants (now colors)
            legend_elements.append(plt.Line2D([0], [0], color='none', label='Variants:'))
            for variant in ['normal', 'nokl', 'conditional']:
                if variant in variant_colors:
                    legend_elements.append(
                        plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=variant_colors[variant], markersize=10,
                                  label=f'  {variant_labels[variant]}', markeredgecolor='white', markeredgewidth=1.0)
                    )
        else:
            # Simple legend for single variant - just methods with different shapes
            standard_color = '#0173B2'  # Use the standard blue
            for method in ['DIM', 'ACE', 'CAA', 'PCA', 'LAT']:
                if method in method_shapes:
                    legend_elements.append(
                        plt.Line2D([0], [0], marker=method_shapes[method], color='w',
                                  markerfacecolor=standard_color, markersize=12,
                                  label=method, markeredgecolor='white', markeredgewidth=1.0)
                    )

        # Single clean legend with larger font
        ax.legend(handles=legend_elements,
                 loc='best',
                 fontsize=14,
                 framealpha=0.95,
                 fancybox=True,
                 shadow=False,
                 borderpad=1,
                 columnspacing=1,
                 handletextpad=0.5)

        # Clean axis labels (no title) with larger fonts
        ax.set_xlabel('Entanglement (%)', fontsize=20)
        ax.set_ylabel('Effectiveness (%)', fontsize=20)

        # Tick label font size
        ax.tick_params(axis='both', which='major', labelsize=16)

        # Improved grid with grey background
        ax.grid(True, alpha=0.5, linestyle='-', linewidth=0.8, color='#DDDDDD')
        ax.set_axisbelow(True)

        # Set axis limits with normal padding
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_padding = (x_max - x_min) * 0.08
        y_padding = (y_max - y_min) * 0.08
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

        # Grey background
        ax.set_facecolor('#f8f9fa')

        # Add axis spines styling
        for spine in ax.spines.values():
            spine.set_edgecolor('#CCCCCC')
            spine.set_linewidth(0.8)

        plt.tight_layout()
        model_name = model.lower().replace(' ', '_').replace('-', '_').replace('.', '_')

        # Save PNG in main directory
        plt.savefig(os.path.join(output_dir, f'pareto_variants_{model_name}.png'),
                   dpi=300, bbox_inches='tight', facecolor='white')

        # Create subdirectories for other formats
        svg_dir = os.path.join(output_dir, 'svgs')
        pdf_dir = os.path.join(output_dir, 'pdfs')
        os.makedirs(svg_dir, exist_ok=True)
        os.makedirs(pdf_dir, exist_ok=True)

        # Save SVG
        plt.savefig(os.path.join(svg_dir, f'pareto_variants_{model_name}.svg'),
                   bbox_inches='tight', facecolor='white')

        # Save PDF
        plt.savefig(os.path.join(pdf_dir, f'pareto_variants_{model_name}.pdf'),
                   bbox_inches='tight', facecolor='white')

        plt.close()

        print(f"Saved improved Pareto frontier with variants for {model} (PNG, SVG, PDF)")


def plot_pareto_frontier(data: Dict, output_dir: str = "."):
    """Create Pareto frontier plots for each model (original version without variants)."""
    summary_df = create_summary_table(data)
    
    if summary_df.empty:
        print("No data available for Pareto frontier plots")
        return
    
    # Filter to only the 5 main methods
    main_methods = ['CAA', 'DIM', 'LAT', 'ACE', 'PCA']
    summary_df = summary_df[summary_df['Method'].isin(main_methods)]
    
    # Create separate individual plots for each model
    models = summary_df['Model'].unique()
    colors = sns.color_palette("husl", len(main_methods))
    method_colors = dict(zip(main_methods, colors))
    
    for model in models:
        model_data = summary_df[summary_df['Model'] == model]
        
        if len(model_data) < 2:
            continue
            
        fig, ax = plt.subplots(figsize=(8, 6))  # Individual plot size
        
        # Calculate average effectiveness and entanglement for each method
        method_averages = model_data.groupby('Method').agg({
            'Effectiveness': 'mean',
            'Entanglement (All)': 'mean'
        }).reset_index()
        
        # Plot points with consistent colors
        for _, row in method_averages.iterrows():
            method = row['Method']
            if method in method_colors:
                ax.scatter(row['Entanglement (All)'], row['Effectiveness'], 
                         c=[method_colors[method]], s=100, alpha=0.7, label=method)
                ax.annotate(method, (row['Entanglement (All)'], row['Effectiveness']), 
                          xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        # Add Pareto frontier
        sorted_data = method_averages.sort_values('Entanglement (All)')
        frontier_points = []
        max_eff_so_far = -float('inf')
        
        for _, row in sorted_data.iterrows():
            if row['Effectiveness'] > max_eff_so_far:
                frontier_points.append((row['Entanglement (All)'], row['Effectiveness']))
                max_eff_so_far = row['Effectiveness']
        
        if len(frontier_points) > 1:
            frontier_x, frontier_y = zip(*frontier_points)
            ax.plot(frontier_x, frontier_y, 'k--', alpha=0.5, linewidth=2, label='Pareto Frontier')
        
        ax.set_xlabel('Entanglement %')
        ax.set_ylabel('Effectiveness %')
        # No title as requested
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        model_name = model.lower().replace(' ', '_').replace('-', '_').replace('.', '_')

        # Create subdirectories for other formats
        svg_dir = os.path.join(output_dir, 'svgs')
        pdf_dir = os.path.join(output_dir, 'pdfs')
        os.makedirs(svg_dir, exist_ok=True)
        os.makedirs(pdf_dir, exist_ok=True)

        # Save in all formats
        plt.savefig(os.path.join(output_dir, f'pareto_frontier_{model_name}.png'),
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(svg_dir, f'pareto_frontier_{model_name}.svg'),
                   bbox_inches='tight')
        plt.savefig(os.path.join(pdf_dir, f'pareto_frontier_{model_name}.pdf'),
                   bbox_inches='tight')
        plt.close()
        
        print(f"Saved individual Pareto frontier plot for {model}")
    
    # Create same-plot version with different colored lines
    fig, ax = plt.subplots(figsize=(12, 6))  # Wider for NeurIPS format
    
    model_colors = {'Qwen': 'blue', 'Llama': 'red'}  # Different colors for models
    model_markers = {'Qwen': 'o', 'Llama': 's'}  # Different markers for models
    
    for model in models:
        model_data = summary_df[summary_df['Model'] == model]
        
        if len(model_data) < 2:
            continue
            
        # Calculate average effectiveness and entanglement for each method
        method_averages = model_data.groupby('Method').agg({
            'Effectiveness': 'mean',
            'Entanglement (All)': 'mean'
        }).reset_index()
        
        # Plot points
        model_color = model_colors.get(model, 'gray')
        model_marker = model_markers.get(model, 'o')
        
        for _, row in method_averages.iterrows():
            method = row['Method']
            ax.scatter(row['Entanglement (All)'], row['Effectiveness'], 
                     c=model_color, s=100, alpha=0.7, marker=model_marker,
                     label=f"{model} - {method}" if model == list(models)[0] else "")
            ax.annotate(f"{method}", (row['Entanglement (All)'], row['Effectiveness']), 
                      xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        # Add Pareto frontier line for this model
        sorted_data = method_averages.sort_values('Entanglement (All)')
        frontier_points = []
        max_eff_so_far = -float('inf')
        
        for _, row in sorted_data.iterrows():
            if row['Effectiveness'] > max_eff_so_far:
                frontier_points.append((row['Entanglement (All)'], row['Effectiveness']))
                max_eff_so_far = row['Effectiveness']
        
        if len(frontier_points) > 1:
            frontier_x, frontier_y = zip(*frontier_points)
            ax.plot(frontier_x, frontier_y, color=model_color, linestyle='--', 
                   alpha=0.7, linewidth=2, label=f'{model} Frontier')
    
    ax.set_xlabel('Entanglement %')
    ax.set_ylabel('Effectiveness %')
    # No title as requested
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()

    # Create subdirectories for other formats
    svg_dir = os.path.join(output_dir, 'svgs')
    pdf_dir = os.path.join(output_dir, 'pdfs')
    os.makedirs(svg_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    # Save in all formats
    plt.savefig(os.path.join(output_dir, 'pareto_frontier_combined.png'),
               dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(svg_dir, 'pareto_frontier_combined.svg'),
               bbox_inches='tight')
    plt.savefig(os.path.join(pdf_dir, 'pareto_frontier_combined.pdf'),
               bbox_inches='tight')
    plt.close()
    
    print("Saved combined Pareto frontier plot")


def print_key_insights(data: Dict):
    """Print key insights from the analysis."""
    summary_df = create_summary_table(data)
    
    if summary_df.empty:
        print("No data available for insights")
        return
    
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    
    # Best performing method overall
    method_avg = summary_df.groupby('Method').agg({
        'Effectiveness': 'mean',
        'Entanglement (All)': 'mean'
    })
    
    best_effectiveness = method_avg['Effectiveness'].idxmax()
    best_low_entanglement = method_avg['Entanglement (All)'].idxmin()
    
    print(f"\n📈 Highest Average Effectiveness: {best_effectiveness} ({method_avg.loc[best_effectiveness, 'Effectiveness']:.1f}%)")
    print(f"🎯 Lowest Average Entanglement: {best_low_entanglement} ({method_avg.loc[best_low_entanglement, 'Entanglement (All)']:.1f}%)")
    
    # Method statistics
    print(f"\n📊 METHOD PERFORMANCE SUMMARY:")
    for method in method_avg.index:
        eff = method_avg.loc[method, 'Effectiveness']
        ent = method_avg.loc[method, 'Entanglement (All)']
        print(f"   {method}: Effectiveness={eff:.1f}%, Entanglement={ent:.1f}%")
    
    # Model comparison if multiple models
    if len(summary_df['Model'].unique()) > 1:
        print(f"\n🤖 MODEL COMPARISON:")
        model_avg = summary_df.groupby('Model')['Effectiveness'].mean()
        for model in model_avg.index:
            print(f"   {model}: Average Effectiveness = {model_avg[model]:.1f}%")
    
    # Concept difficulty (lower effectiveness = harder)
    if len(summary_df['Concept'].unique()) > 1:
        print(f"\n🎯 CONCEPT DIFFICULTY (by average effectiveness):")
        concept_avg = summary_df.groupby('Concept')['Effectiveness'].mean().sort_values(ascending=False)
        for concept in concept_avg.index:
            print(f"   {concept}: {concept_avg[concept]:.1f}%")
    
    print("\n" + "="*60)


def run_analysis(search_dirs: List[str] = None, output_prefix: str = "concept_metrics"):
    """Run the complete analysis with specified search directories."""
    print("Analyzing concept metrics results...")
    
    # Collect all metrics files
    metrics_files = collect_concept_metrics_files(search_dirs)
    if not metrics_files:
        print("No concept metrics files found!")
        return
    
    # Load all data
    data = load_metrics_data(metrics_files)
    
    if not data:
        print("No valid data loaded!")
        return
    
    # Print key insights
    print_key_insights(data)
    
    # Save all tables
    print("\nSaving tables and analysis...")
    save_tables_to_files(data, output_prefix)
    
    # Create visualizations
    print("\nCreating visualizations...")
    output_dir = "analysis_results_arxiv"
    os.makedirs(output_dir, exist_ok=True)
    # create_visualization_plots(data, output_dir=output_dir)
    plot_pareto_frontier_with_variants(data, output_dir=output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"📊 Summary data: {output_prefix}_summary.csv")
    print(f"📈 Analysis report: {output_prefix}_analysis.md")
    print(f"📉 Effectiveness table: {output_prefix}_effectiveness.csv")
    print(f"🎯 Entanglement tables: {output_prefix}_entanglement_*.csv")
    print(f"📊 Method comparison: {output_prefix}_method_comparison.csv")
    print(f"📸 Plots saved to: {output_dir}/")
    print(f"   - pareto_frontier_*.png")
    print(f"   - *_SteerCompare.png (from generate_all_plots.py)")
    print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze concept metrics and create comprehensive reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analyze_concept_metrics.py
    python analyze_concept_metrics.py --experiments-dir experiments
    python analyze_concept_metrics.py --output-prefix my_analysis
        """
    )
    
    parser.add_argument('--experiments-dir', default='experiments',
                       help='Directory containing experiment results (default: experiments)')
    parser.add_argument('--output-prefix', default='concept_metrics_analysis',
                       help='Prefix for output files (default: concept_metrics_analysis)')
    
    args = parser.parse_args()
    
    search_dirs = [args.experiments_dir]
    run_analysis(search_dirs, args.output_prefix)


if __name__ == '__main__':
    main()
