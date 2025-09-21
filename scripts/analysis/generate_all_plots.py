#!/usr/bin/env python3
"""
Generate heatmap plots for all models using the provided plotting function.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm, ListedColormap
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# Add necessary directories to Python path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'run'))

from analyze_concept_metrics import parse_experiment_name
from run_concept_metrics import load_all_scores_from_experiment
from run_full_pipeline import find_baseline_experiments
import os
import yaml
import glob
from typing import Dict, List, Optional
from collections import defaultdict


def create_heatmap_dataframes(experiments_dir: str) -> Dict[str, pd.DataFrame]:
    """Create DataFrames for heatmap plotting, organized by model."""
    
    # Use same logic as run_full_pipeline.py for finding experiments
    # Find all non-baseline experiment directories
    pattern = os.path.join(experiments_dir, '*', '*')
    all_exp_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    steering_dirs = [d for d in all_exp_dirs if not os.path.basename(os.path.dirname(d)).startswith('baseline_')]
    
    # Find baseline directories using existing function
    baseline_pattern = os.path.join(experiments_dir, 'baseline_*', '*')
    baseline_dirs = [d for d in glob.glob(baseline_pattern) if os.path.isdir(d)]
    
    # Organize data by model
    model_data = defaultdict(list)
    
    # Process steering experiments
    for exp_dir in steering_dirs:
        exp_dir_file = os.path.join(exp_dir, "concept_metrics.yaml")  # for continuity with parse_experiment_name
        try:
            method, concept, model = parse_experiment_name(exp_dir_file)
            method = method.upper()
        except (ValueError, TypeError) as e:
            print(f"Skipping experiment {exp_dir}: {e}")
            continue
        
        # Load all scores from this experiment
        scores = load_all_scores_from_experiment(exp_dir)
        
        if not scores:
            continue
            
        # Add each dataset score as a row
        for dataset, score in scores.items():
            model_data[model].append({
                'steering_method': method,
                'concept': concept,
                'dataset': dataset,
                'avg_metric': score,
                'experiment_dir': exp_dir
            })
    
    # Process baseline experiments
    for exp_dir in baseline_dirs:
        exp_name = os.path.basename(os.path.dirname(exp_dir))
        parts = exp_name.split('_')
        if len(parts) >= 2 and parts[0] == 'baseline':
            model = parts[1]
            
            # Load all scores from this baseline
            scores = load_all_scores_from_experiment(exp_dir)
            
            if not scores:
                continue
                
            # Add each dataset score as a row with method 'base'
            for dataset, score in scores.items():
                model_data[model].append({
                    'steering_method': 'base',
                    'concept': 'baseline',
                    'dataset': dataset,
                    'avg_metric': score,
                    'experiment_dir': exp_dir
                })
    
    # Convert to DataFrames
    model_dfs = {}
    for model, data in model_data.items():
        if data:  # Only create DataFrame if we have data
            model_dfs[model] = pd.DataFrame(data)
    
    return model_dfs


def generate_plot(name, df, heatmap_type='normal', exclude_methods=None):
    # Set style
    sns.set(style="whitegrid")
    
    # Filter out excluded methods
    if exclude_methods is None:
        exclude_methods = []  # 'DIM_NOKL']  # Default exclusions
    
    df_filtered = df[~df['steering_method'].isin(exclude_methods)].copy()
    
    # Get actual available methods from the data
    all_available_methods = [m for m in df_filtered['steering_method'].unique() if m != 'base']
    
    # Filter methods based on heatmap type
    base_methods = ['DIM', 'ACE', 'CAA', 'PCA', 'LAT']

    if heatmap_type == 'normal':
        # Only main methods (no variants)
        available_methods = [m for m in base_methods if m in all_available_methods]
        suffix = ""
    elif heatmap_type == 'nokl':
        # Only NOKL variants
        nokl_methods = [f"{m}_NOKL" for m in base_methods]
        available_methods = [m for m in nokl_methods if m in all_available_methods]
        suffix = "_nokl"
    elif heatmap_type == 'conditional':
        # Only CONDITIONAL variants
        conditional_methods = [f"{m}_CONDITIONAL" for m in base_methods]
        available_methods = [m for m in conditional_methods if m in all_available_methods]
        suffix = "_conditional"
    elif heatmap_type == 'cosmic':
        # Only COSMIC variants
        cosmic_methods = [f"{m}_COSMIC" for m in base_methods]
        available_methods = [m for m in cosmic_methods if m in all_available_methods]
        suffix = "_cosmic"
    else:
        raise ValueError(f"Unknown heatmap_type: {heatmap_type}")

    # Filter the dataframe to only include the methods for this heatmap type
    df_filtered = df_filtered[df_filtered['steering_method'].isin(available_methods + ['base'])].copy()

    # Order methods consistently: DIM, ACE, CAA, PCA, LAT (in whatever variant)
    ordered_methods = []
    for base_method in base_methods:
        for method in available_methods:
            if method.startswith(base_method):
                ordered_methods.append(method)
                break
    available_methods = ordered_methods
    
    print(f"Available methods for {name}: {available_methods}")
    
    # Define targets and metrics
    targets = ["Harmful Generation", "Hallucination Extrinsic", "Hallucination Intrinsic", "Explicit Bias", "Implicit Bias"]
    
    metrics = [
        "SALAD ASR", "PreciseWikiQA Halluc.",
        "FaithEvalCounterFac.", "FaithEvalInconsis.", "FaithEvalUnanswer.",
        "ToxiGen Acc.", "BBQ Acc.", "Brand Bias", "Sycophancy",
        "Anthropomorphism", "User Retention", "GPQA", "ARC-C",
        "TruthfulQA", "Sneaking",
        "Commonsense Morality", "Political Views"
    ]
    
    # NOTE: due to its variability, we did not include Twinviews in the aggregations for Llama due to known issues with formatting and substring matching making it hard--and potentially misleading--to interpret.
    if 'llama' in name.lower() and 'Twinviews' in df_filtered['dataset'].values:
        df_filtered = df_filtered[df_filtered['dataset'] != 'Twinviews'].copy()
        metrics = [m for m in metrics if m != 'Political Views']
        print(f"DEBUG: Auto-excluding Twinviews data rows for Llama heatmap")
    
    
    # Dynamically create data structure based on available methods
    data_dict = {}
    for target in targets:
        for method in available_methods:
            data_dict[f"{target} - {method}"] = [0] * len(metrics)
    
    # Map concepts to target behaviors (including refusal as harmful generation)
    concept_to_target = {
        "Refusal Base": "Harmful Generation",
        "Nokl Refusal Base": "Harmful Generation", 
        "Hallucination Extrinsic": "Hallucination Extrinsic",
        "Hallucination Intrinsic": "Hallucination Intrinsic", 
        "Explicit Bias": "Explicit Bias",
        "Implicit Bias": "Implicit Bias",
        "Nokl Explicit Bias": "Explicit Bias",
        "Nokl Implicit Bias": "Implicit Bias",
        "Nokl Hallucination Extrinsic": "Hallucination Extrinsic",
        "Nokl Hallucination Intrinsic": "Hallucination Intrinsic",
        # Underscore names too
        "refusal_base": "Harmful Generation",
        "hallucination_extrinsic": "Hallucination Extrinsic",
        "hallucination_intrinsic": "Hallucination Intrinsic", 
        "explicit_bias": "Explicit Bias",
        "implicit_bias": "Implicit Bias"
    }
    
    task_to_metric = {
        "SaladBench": "SALAD ASR",
        "PreciseWiki": "PreciseWikiQA Halluc.",  # or .Halluc. depending on which metric you want to fill
        "FaithEvalCounterfactual": "FaithEvalCounterFac.",
        "FaithEvalInconsistent": "FaithEvalInconsis.",
        "FaithEvalUnanswerable": "FaithEvalUnanswer.",
        "BBQ": "BBQ Acc.",
        "ToxiGen": "ToxiGen Acc.", 
        "DarkBenchBrandBias": "Brand Bias",
        "DarkBenchSynchopancy": "Sycophancy",
        "DarkBenchAnthro": "Anthropomorphism",
        "DarkBenchRetention": "User Retention",
        "DarkBenchSneaking": "Sneaking",
        "GPQA": "GPQA",
        "ARC_C": "ARC-C",
        "TruthfulQA": "TruthfulQA", 
        "CMTEST": "Commonsense Morality",
        "Twinviews": "Political Views"
        
    }

    relative_data_manual = pd.DataFrame.from_dict(data_dict, orient="index", columns=metrics)
    
    # Add base model - include variant symbol if applicable
    variant_symbol = ""
    if heatmap_type == 'nokl':
        variant_symbol = r"$^{\ddagger}$"  # double dagger for NoKL
    elif heatmap_type == 'conditional':
        variant_symbol = r"$^{\dagger}$"   # dagger for conditional
    elif heatmap_type == 'cosmic':
        variant_symbol = r"$^{*}$"         # asterisk for cosmic

    base_model_label = f"Base {name.title()}{variant_symbol}"
    base_model_df = pd.DataFrame([[0.0] * len(metrics)], index=[base_model_label], columns=metrics)
    
    # Populate data from the DataFrame
    for i, row in df_filtered.iterrows():
        metric = row["avg_metric"]
        metric_col = task_to_metric.get(row["dataset"])
            
        if "base" in row["steering_method"]:
            # Add to baseline
            if metric_col in base_model_df.columns:
                base_model_df.at[base_model_label, metric_col] = metric * 100
        else: 
            # Map concept to target behavior
            target = concept_to_target.get(row["concept"])
            if target and metric_col:
                row_key = f"{target} - {row['steering_method']}"
                if row_key in relative_data_manual.index and metric_col in relative_data_manual.columns:
                    relative_data_manual.at[row_key, metric_col] = metric * 100
    
    for i, row in relative_data_manual.iterrows():
        for j in range(len(row)):
            metric_col = metrics[j]
            
            if metric_col in base_model_df.columns and row[metric_col] != 0:
                base_value = base_model_df.at[base_model_label, metric_col]
                relative_data_manual.at[i, metric_col] = row[metric_col] - base_value
                base_model_df.at[base_model_label, metric_col] = base_model_df.at[base_model_label, metric_col]
    
    
    # Row groupings - only include targets that have data
    row_groups = []
    section_titles = []
    
    for target in targets:
        target_rows = [i for i in relative_data_manual.index if i.startswith(target)]
        if target_rows:  # Only include if we have data
            row_groups.append(target_rows)
            section_titles.append(f"Target Behavior: {target.replace('Harmful Generation', 'Harmful Generation (Refusal)')}")
    
    # Create custom height ratios - adjust based on number of sections
    num_sections = len(row_groups) + 1  # +1 for baseline
    num_methods = len(available_methods)
    
    # Adjust sizing based on heatmap type
    # Since we're showing separate heatmaps for each variant, keep standard sizing
    method_height = max(1, num_methods // 2)  # Standard sizing
    fig = plt.figure(figsize=(12, max(6, num_sections * 1.5)))  # Standard figure
    
    height_ratios = [1] + [method_height] * len(row_groups)  # baseline + target sections
    gs = GridSpec(nrows=num_sections, ncols=1, height_ratios=height_ratios)
    axes = [fig.add_subplot(gs[i]) for i in range(num_sections)]
    
    # Colormap and normalization
    pub_cmap = matplotlib.colormaps['RdBu']
    norm = TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20)
    
    # Create sections dynamically
    sections = [(base_model_label, base_model_df)]
    for i, (title, rows) in enumerate(zip(section_titles, row_groups)):
        sections.append((title, relative_data_manual.loc[rows]))
        
    # Plot each section
    for idx, (ax, (title, section_df)) in enumerate(zip(axes, sections)):
        cmap = ListedColormap(["#d3d3d3"]) if title == base_model_label else pub_cmap
        show_x = (idx == len(axes) - 1)
    
        # Determine which metrics are primary for this target
        if "Harmful Generation" in title or "Refusal" in title:
            primary_metrics = ["SALAD ASR"]
        elif "Extrinsic Hallucination" in title or "Hallucination Extrinsic" in title:
            primary_metrics = ["PreciseWikiQA Halluc."]
        elif "Intrinsic Hallucination" in title or "Hallucination Intrinsic" in title:
            primary_metrics = ["FaithEvalCounterFac.", "FaithEvalInconsis.", "FaithEvalUnanswer."]
        elif "Explicit Bias" in title:
            primary_metrics = ["ToxiGen Acc."]
        elif "Implicit Bias" in title:
            primary_metrics = ["BBQ Acc."]
        else:
            primary_metrics = []
        
        # Format annotations, bolding primary metric values
        annot_matrix = section_df.copy().astype(str)
        for r in section_df.index:
            for c in section_df.columns:
                val = section_df.at[r, c]
                # Use 1 decimal place for all values in heatmap
                rounded_val = f"{val:.1f}"
                
                if c in primary_metrics:
                    rounded_val = f"{rounded_val}*"    
                annot_matrix.at[r, c] = f"{rounded_val}"

        heatmap = sns.heatmap(
            section_df,
            fmt="",
            annot=annot_matrix,
            linewidths=0.2,
            cmap=cmap,
            center=0,
            vmin=-20, vmax=20,
            norm=norm if title != base_model_label else None,
            cbar=False,
            xticklabels=True if title == base_model_label else False,
            yticklabels=True,
            ax=ax
        )

        # Draw thick rectangle around primary metric columns
        for i, row_name in enumerate(section_df.index):
            for j, col_name in enumerate(section_df.columns):
                if col_name in primary_metrics:
                    ax.add_patch(plt.Rectangle(
                        (j, i), 1, 1,
                        fill=False,
                        edgecolor='black',
                        linewidth=1,
                        clip_on=False,
                        zorder=10  # Ensure it's above heatmap
                    ))

            
        # Y-axis labels with nice method formatting
        def format_method_name(method_name):
            """Format method names with subscripts for variants."""
            if method_name == base_model_label:
                if "qwen" in name.lower():
                    # Extract size info if available, otherwise default
                    if "7b" in name.lower():
                        return "Qwen-2.5\n7B-Instruct"
                    elif "1_5b" in name.lower() or "1.5b" in name.lower():
                        return "Qwen-2.5\n1.5B-Instruct"
                    elif "3b" in name.lower():
                        return "Qwen-2.5\n3B-Instruct"
                    else:
                        return "Qwen-2.5\nInstruct"
                elif "llama" in name.lower():
                    if "8b" in name.lower():
                        return "Llama-3.1\n8B-Instruct"
                    else:
                        return "Llama-3.1\nInstruct"
                else:
                    return f"{name.title()}\nModel"
            
            # Handle method variants with superscript symbols
            method_mappings = {
                # Conditional variants (dagger)
                'CAA_CONDITIONAL': r'$\mathbf{CAA^{\dagger}}$',
                'DIM_CONDITIONAL': r'$\mathbf{DIM^{\dagger}}$',
                'LAT_CONDITIONAL': r'$\mathbf{LAT^{\dagger}}$',
                'ACE_CONDITIONAL': r'$\mathbf{ACE^{\dagger}}$',
                'PCA_CONDITIONAL': r'$\mathbf{PCA^{\dagger}}$',
                # NoKL variants (double dagger)
                'DIM_NOKL': r'$\mathbf{DIM^{\ddagger}}$',
                'ACE_NOKL': r'$\mathbf{ACE^{\ddagger}}$',
                'CAA_NOKL': r'$\mathbf{CAA^{\ddagger}}$',
                'PCA_NOKL': r'$\mathbf{PCA^{\ddagger}}$',
                'LAT_NOKL': r'$\mathbf{LAT^{\ddagger}}$',
                # Cosmic variants (asterisk)
                'ACE_COSMIC': r'$\mathbf{ACE^{*}}$',
                'CAA_COSMIC': r'$\mathbf{CAA^{*}}$',
                'DIM_COSMIC': r'$\mathbf{DIM^{*}}$',
                'PCA_COSMIC': r'$\mathbf{PCA^{*}}$',
                'LAT_COSMIC': r'$\mathbf{LAT^{*}}$'
            }
            
            return method_mappings.get(method_name, f'$\\mathbf{{{method_name}}}$')
        
        new_labels = [
            format_method_name(idx if idx == base_model_label else idx.split(" - ")[-1])
            for idx in section_df.index
        ]
        ax.set_yticklabels(new_labels, rotation=0, fontsize=10 if title == base_model_label else 12)
    
        # Titles and tick formatting
        ax.set_title(title if title != base_model_label else None, fontsize=11, weight='bold', pad=0)

        if title == base_model_label:
            ax.xaxis.set_ticks_position("top")
            ax.xaxis.set_label_position("top")
            ax.tick_params(axis='x', labelsize=10)
            ax.set_xticklabels(
                ax.get_xticklabels(),
                rotation=30,
                ha='left',                # Anchor to right
                rotation_mode='anchor'    # Rotate from anchor instead of center
            )
    
    
    # Add unified horizontal colorbar
    cax = inset_axes(
        parent_axes=axes[-1],
        width="30%",     # Width of colorbar relative to parent
        height="2.5%",   # Height of colorbar
        loc='lower right',
        bbox_to_anchor=(-0.1, 0.05, 1, 1),
        bbox_transform=fig.transFigure,
        borderpad=0,
    )
    
    cb = fig.colorbar(
        plt.cm.ScalarMappable(cmap=pub_cmap, norm=norm),
        cax=cax,
        orientation='horizontal'  # ⬅️ Make it horizontal
    )
    
    # Style tick labels
    cb.ax.tick_params(labelsize=14, direction='out')
    for tick in cb.ax.get_xticklabels():  # ⬅️ horizontal = xticks
        tick.set_fontweight('bold')
        tick.set_fontsize(14)


    cb.set_label("Absolute % Change", fontsize=14, weight='bold', labelpad=10)

    # Add legend for variant symbols if applicable
    if heatmap_type != 'normal':
        legend_text = ""
        if heatmap_type == 'nokl':
            legend_text = r"$^{\ddagger}$ No KL Divergence Check"
        elif heatmap_type == 'conditional':
            legend_text = r"$^{\dagger}$ With CAST for Conditional Application"
        elif heatmap_type == 'cosmic':
            legend_text = r"$^{*}$ With COSMIC for Concept-Specific Steering"

        # Add text at bottom left
        fig.text(0.02, 0.02, legend_text, fontsize=12, weight='bold',
                transform=fig.transFigure, verticalalignment='bottom')

    # Create output directory and subdirectories
    output_dir = "analysis_results_arxiv"
    svg_dir = os.path.join(output_dir, 'svgs')
    pdf_dir = os.path.join(output_dir, 'pdfs')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    # Save in all formats
    filename = f"{name}{suffix}_SteerCompare"
    png_path = os.path.join(output_dir, f"{filename}.png")
    svg_path = os.path.join(svg_dir, f"{filename}.svg")
    pdf_path = os.path.join(pdf_dir, f"{filename}.pdf")

    plt.savefig(png_path, dpi=600, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')

    print(f"Saved plot to: {png_path} (+ SVG and PDF versions)")
    plt.show()


def analyze_missing_data(experiments_dir: str, exclude_methods=None):
    """Analyze and report missing datasets/experiments."""
    if exclude_methods is None:
        exclude_methods = ['PCA_COSMIC', 'CAA_COSMIC']
    
    # Expected combinations
    expected_methods = ['ACE', 'CAA', 'DIM', 'LAT', 'PCA']
    expected_methods = [m for m in expected_methods if m not in exclude_methods]
    
    expected_concepts = ['Refusal Base', 'Explicit Bias', 'Implicit Bias', 'Hallucination Extrinsic', 'Hallucination Intrinsic']
    
    expected_datasets = {
        'SaladBench', 'PreciseWiki', 'FaithEvalCounterfactual', 'FaithEvalInconsistent', 
        'FaithEvalUnanswerable', 'ToxiGen', 'BBQ', 'DarkBenchBrandBias', 'DarkBenchSynchopancy', 
        'DarkBenchAnthro', 'DarkBenchRetention', 'DarkBenchSneaking', 'GPQA', 'ARC_C', 
        'TruthfulQA', 'CMTEST', 'Twinviews'
    }
    
    # Load actual data
    model_dfs = create_heatmap_dataframes(experiments_dir)
    
    print("\n" + "="*80)
    print("MISSING DATA ANALYSIS")
    print("="*80)
    
    for model, df in model_dfs.items():
        print(f"\n📊 MODEL: {model.upper()}")
        print("-" * 50)
        
        # Filter out excluded methods
        df_filtered = df[~df['steering_method'].isin(exclude_methods)].copy()
        
        # Get actual data
        actual_methods = set(df_filtered[df_filtered['steering_method'] != 'base']['steering_method'].unique())
        actual_concepts = set(df_filtered['concept'].unique()) - {'Baseline'}
        actual_datasets = set(df_filtered['dataset'].unique())
        baseline_datasets = set(df_filtered[df_filtered['steering_method'] == 'base']['dataset'].unique())
        
        print(f"✅ Found methods: {sorted(actual_methods)}")
        print(f"✅ Found concepts: {sorted(actual_concepts)}")
        print(f"✅ Found datasets: {len(actual_datasets)} total")
        print(f"✅ Baseline datasets: {len(baseline_datasets)} total")
        
        # Missing methods
        missing_methods = set(expected_methods) - actual_methods
        if missing_methods:
            print(f"\n❌ Missing methods: {sorted(missing_methods)}")
        
        # Missing concepts  
        missing_concepts = set(expected_concepts) - actual_concepts
        if missing_concepts:
            print(f"❌ Missing concepts: {sorted(missing_concepts)}")
            
        # Missing datasets
        missing_datasets = expected_datasets - actual_datasets
        if missing_datasets:
            print(f"❌ Missing datasets: {sorted(missing_datasets)}")
            
        # Check baseline coverage
        missing_baseline = expected_datasets - baseline_datasets
        if missing_baseline:
            print(f"❌ Missing baseline datasets: {sorted(missing_baseline)}")
        
        # Detailed method-concept combinations
        print(f"\n🔍 METHOD-CONCEPT COMBINATIONS:")
        for method in sorted(actual_methods):
            method_concepts = set(df_filtered[df_filtered['steering_method'] == method]['concept'].unique())
            missing_for_method = set(expected_concepts) - method_concepts
            if missing_for_method:
                print(f"  {method}: Missing {sorted(missing_for_method)}")
            else:
                print(f"  {method}: ✅ Complete")
        
        # Dataset coverage by concept
        print(f"\n🎯 DATASET COVERAGE BY CONCEPT:")
        for concept in sorted(actual_concepts):
            concept_datasets = set(df_filtered[df_filtered['concept'] == concept]['dataset'].unique())
            concept_methods = set(df_filtered[df_filtered['concept'] == concept]['steering_method'].unique())
            
            if concept == 'Refusal Base':
                key_datasets = {'SaladBench'}
            elif 'Hallucination Extrinsic' in concept:
                key_datasets = {'PreciseWiki'}
            elif 'Hallucination Intrinsic' in concept:
                key_datasets = {'FaithEvalCounterfactual', 'FaithEvalInconsistent', 'FaithEvalUnanswerable'}
            elif 'Explicit Bias' in concept:
                key_datasets = {'ToxiGen'}
            elif 'Implicit Bias' in concept:
                key_datasets = {'BBQ'}
            else:
                key_datasets = set()
            
            missing_key = key_datasets - concept_datasets
            if missing_key:
                print(f"  {concept}: ❌ Missing key datasets {sorted(missing_key)} for methods {sorted(concept_methods)}")
            else:
                print(f"  {concept}: ✅ Has key datasets for methods {sorted(concept_methods)}")


def generate_all_plots(experiments_dir: str):
    """Generate plots for all models in the experiments directory."""
    
    # First analyze missing data
    exclude_methods = ['PCA_COSMIC', 'CAA_COSMIC', 'PCA_NEW_COSMIC']
    analyze_missing_data(experiments_dir, exclude_methods)
    
    # Create DataFrames for each model
    model_dfs = create_heatmap_dataframes(experiments_dir)
    
    if not model_dfs:
        print("No model data found!")
        return
    
    print(f"\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)

    # Check if we have variant methods
    all_methods = set()
    for df in model_dfs.values():
        all_methods.update(df['steering_method'].unique())

    has_nokl = any('_NOKL' in m for m in all_methods)
    has_conditional = any('_CONDITIONAL' in m for m in all_methods)
    has_cosmic = any('_COSMIC' in m for m in all_methods)

    # Determine which heatmaps to generate
    heatmap_types = ['normal']
    if has_nokl:
        heatmap_types.append('nokl')
    if has_conditional:
        heatmap_types.append('conditional')
    if has_cosmic:
        heatmap_types.append('cosmic')

    print(f"Generating {len(heatmap_types)} heatmap types: {heatmap_types}")

    # Generate plots for each model and each variant type
    for model, df in model_dfs.items():
        # Map model names to more readable format - keep model size info
        model_name = model.lower().replace('-', '_')

        # Clean up version numbers but keep size
        if 'llama' in model_name:
            # e.g., llama3-1-8b -> llama3_1_8b
            model_name = model_name.replace('llama3.1', 'llama3_1')
        elif 'qwen' in model_name:
            # e.g., qwen25-7b -> qwen25_7b
            model_name = model_name.replace('qwen2.5', 'qwen25')

        # Check which variant types this specific model has
        model_methods = set(df['steering_method'].unique())
        model_has_nokl = any('_NOKL' in m for m in model_methods)
        model_has_conditional = any('_CONDITIONAL' in m for m in model_methods)
        model_has_cosmic = any('_COSMIC' in m for m in model_methods)

        # Determine heatmaps to generate for this specific model
        model_heatmap_types = ['normal']
        if model_has_nokl:
            model_heatmap_types.append('nokl')
        if model_has_conditional:
            model_heatmap_types.append('conditional')
        if model_has_cosmic:
            model_heatmap_types.append('cosmic')

        for heatmap_type in model_heatmap_types:
            print(f"\n📈 Generating {heatmap_type} heatmap for {model}...")

            try:
                generate_plot(model_name, df, heatmap_type=heatmap_type, exclude_methods=exclude_methods)
                print(f"✅ {heatmap_type.capitalize()} heatmap saved for {model}")
            except Exception as e:
                print(f"❌ Error generating {heatmap_type} heatmap for {model}: {e}")
                continue


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python generate_all_plots.py <experiments_dir>")
        print("Example: python generate_all_plots.py experiments/")
        sys.exit(1)
    
    experiments_dir = sys.argv[1]
    if not os.path.exists(experiments_dir):
        print(f"Directory not found: {experiments_dir}")
        sys.exit(1)
    
    generate_all_plots(experiments_dir)
