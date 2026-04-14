"""
Visualizations for the attention pipeline.
This module will contain functions for generating plots, heatmaps, etc.
"""

from typing import Dict, Any, List
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math

logging.getLogger("matplotlib.category").setLevel(logging.WARNING)

def get_model_family(model_name: str) -> str:
    """Helper to categorize model names into families."""
    name_lower = model_name.lower()
    if "claude" in name_lower:
        return "Claude"
    elif "gpt" in name_lower:
        return "GPT"
    elif "gemini" in name_lower:
        return "Gemini"
    elif "qwen" in name_lower:
        return "Qwen"
    elif "deepseek" in name_lower:
        return "DeepSeek"
    elif "glm" in name_lower:
        return "GLM"
    else:
        return "Other"

def generate_visuals(result: Dict[str, Any], output_dir: str) -> None:
    """
    Generate plots and heatmaps based on the analysis results.
    Currently empty, to be implemented later.
    """
    pass

def generate_cross_model_visuals(all_results: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Generate plots comparing all models.
    """
    if not all_results:
        return

    # Extract PSS scores and prepare dataframe
    data = []
    for res in all_results:
        model_name = res["model"]
        pss = res["sensitivity"].get("perturbation_sensitivity_score", 0)
        family = get_model_family(model_name)
        data.append({"Model": model_name, "PSS": pss, "Family": family})
    
    df = pd.DataFrame(data)
    # Sort from lowest to highest PSS (most invariant to least invariant), then alphabetically by model
    df = df.sort_values(by=["PSS", "Model"], ascending=[True, True]).reset_index(drop=True)
    
    # 1. PSS Ranking Bar Chart
    plt.figure(figsize=(10, 8))
    
    # Define color palette for families
    families = df["Family"].unique()
    palette = sns.color_palette("Set2", n_colors=len(families))
    color_map = dict(zip(families, palette))
    
    ax = sns.barplot(
        x="PSS", 
        y="Model", 
        hue="Family", 
        data=df, 
        dodge=False, 
        palette=color_map,
        zorder=3
    )
    
    plt.title("Model Ranking by Perturbation Sensitivity Score (PSS)\n(Lower is better / more invariant)", pad=15)
    plt.xlabel("Perturbation Sensitivity Score")
    plt.ylabel("Model")
    plt.grid(axis='x', linestyle='--', alpha=0.7, zorder=0)
    
    # Annotate best and worst
    if not df.empty:
        best_model = df.iloc[0]
        worst_model = df.iloc[-1]
        
        # Best
        ax.text(
            best_model["PSS"], 
            0, 
            f'  Best: {best_model["PSS"]:.3f}', 
            va='center', 
            ha='left', 
            fontsize=10,
            fontweight='bold',
            color='green'
        )
        
        # Worst
        ax.text(
            worst_model["PSS"], 
            len(df) - 1, 
            f'  Worst: {worst_model["PSS"]:.3f}', 
            va='center', 
            ha='left', 
            fontsize=10,
            fontweight='bold',
            color='red'
        )
        
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "pss_ranking_bar_chart.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 2. ESI 2->3 Under-Triage Event Count Bar Chart
    triage_data = []
    
    for res in all_results:
        model_name = res["model"]
        transitions = res.get("dangerous_transitions", [])
        
        counts = {
            "female_2_3": 0,
            "male_2_3": 0,
            "nb_2_3": 0,
            "critical_2_5": 0,
        }
        
        for t in transitions:
            if t["baseline_esi"] == 2 and t["shifted_esi"] == 3:
                if t["sex_label"] == "female":
                    counts["female_2_3"] += t["count"]
                elif t["sex_label"] == "male":
                    counts["male_2_3"] += t["count"]
                elif t["sex_label"] == "nb_label_only":
                    counts["nb_2_3"] += t["count"]
            elif t["baseline_esi"] == 2 and t["shifted_esi"] == 5:
                counts["critical_2_5"] += t["count"]
                
        total_2_3 = counts["female_2_3"] + counts["male_2_3"] + counts["nb_2_3"]
        
        triage_data.append({
            "Model": model_name,
            "Female (2→3)": counts["female_2_3"],
            "Male (2→3)": counts["male_2_3"],
            "Non-binary (2→3)": counts["nb_2_3"],
            "Critical 2→5": counts["critical_2_5"],
            "Total 2→3": total_2_3
        })
        
    df_triage = pd.DataFrame(triage_data)
    df_triage = df_triage.sort_values(["Total 2→3", "Model"], ascending=[True, True]).reset_index(drop=True)
    
    plt.figure(figsize=(10, 8))
    
    models = df_triage["Model"]
    
    p1 = plt.barh(models, df_triage["Female (2→3)"], color='lightcoral', label='Female (2→3)', edgecolor='white', zorder=3)
    p2 = plt.barh(models, df_triage["Male (2→3)"], left=df_triage["Female (2→3)"], color='cornflowerblue', label='Male (2→3)', edgecolor='white', zorder=3)
    p3 = plt.barh(models, df_triage["Non-binary (2→3)"], left=df_triage["Female (2→3)"] + df_triage["Male (2→3)"], color='mediumseagreen', label='Non-binary (2→3)', edgecolor='white', zorder=3)
    
    has_critical = df_triage[df_triage["Critical 2→5"] > 0]
    if not has_critical.empty:
        for idx, row in has_critical.iterrows():
            x_pos = row["Total 2→3"] + 1
            plt.scatter(x_pos, idx, color='red', marker='*', s=150, zorder=5, label='CRITICAL 2→5 (Undertriage)' if idx == has_critical.index[0] else "")
            plt.text(x_pos + 0.5, idx, f'{int(row["Critical 2→5"])}x 2→5', va='center', color='red', fontweight='bold', zorder=5)
            
    plt.title("ESI 2→3 Under-Triage Event Count by Sex Label", pad=15)
    plt.xlabel("Count of ESI 2→3 Under-Triage Events")
    plt.ylabel("Model")
    plt.grid(axis='x', linestyle='--', alpha=0.7, zorder=0)
    
    # Fix legend
    handles, labels = plt.gca().get_legend_handles_labels()
    # Deduplicate labels just in case
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower right')
    
    plt.tight_layout()
    out_path_triage = os.path.join(output_dir, "esi_2_to_3_undertriage_stacked_bar.png")
    plt.savefig(out_path_triage, dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Four-Layer Effect Decomposition Heatmap
    layer_names = ["Presence", "Category", "Gender Value", "NB Token"]
    layer_keys = [
        "L1_presence_abs_mean_effect", 
        "L2_binary_vs_nb_abs_mean", 
        "L3_female_vs_male_abs_mean", 
        "L4_nb_token_abs_mean"
    ]
    p_keys = [
        "L1_wilcoxon_p",
        "L2_wilcoxon_p",
        "L3_wilcoxon_p",
        "L4_wilcoxon_p"
    ]

    heatmap_data = []
    annot_data = []
    
    # We use df which is already sorted by PSS from lowest to highest
    sorted_models = df["Model"].tolist()
    
    for model_name in sorted_models:
        # Find the result for this model
        res = next((r for r in all_results if r["model"] == model_name), None)
        if not res or "effect_decomposition" not in res:
            heatmap_data.append([0.0]*4)
            annot_data.append([""]*4)
            continue
            
        ed = res["effect_decomposition"]
        
        row_vals = []
        row_annots = []
        for i, key in enumerate(layer_keys):
            val = ed.get(key, 0.0)
            row_vals.append(val)
            
            p_val = ed.get(p_keys[i])
            # Construct annotation text: value with an asterisk if significant
            annot_str = f"{val:.3f}"
            if p_val is not None and p_val < 0.05:
                annot_str += "*"
            row_annots.append(annot_str)
            
        heatmap_data.append(row_vals)
        annot_data.append(row_annots)
        
    df_hm = pd.DataFrame(heatmap_data, index=sorted_models, columns=layer_names)
    df_annot = pd.DataFrame(annot_data, index=sorted_models, columns=layer_names)

    plt.figure(figsize=(8, 10))
    sns.heatmap(
        df_hm, 
        annot=df_annot, 
        fmt="", 
        cmap="YlOrRd", 
        cbar_kws={'label': 'Absolute Mean Effect (ESI Shift)'},
        linewidths=.5
    )
    plt.title("Four-Layer Effect Decomposition by Model\n(* denotes p < 0.05)", pad=15)
    plt.ylabel("Model (sorted by PSS)")
    plt.xlabel("Decomposition Layer")
    plt.tight_layout()
    
    out_path_hm = os.path.join(output_dir, "four_layer_decomposition_heatmap.png")
    plt.savefig(out_path_hm, dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Accuracy by Condition Grouped Bar Chart
    acc_data = []
    
    for res in all_results:
        model_name = res["model"]
        ed = res.get("effect_decomposition", {})
        
        # Extract accuracies from the decomposition stats
        # They were computed inside decompose_sex_info_effect
        acc_male = ed.get("L3_accuracy_male", 0.0)
        acc_female = ed.get("L3_accuracy_female", 0.0)
        acc_nb = ed.get("L4_accuracy_nonbinary", 0.0)
        acc_no_label = ed.get("L1_accuracy_without_sex_info", 0.0)
        
        acc_data.append({
            "Model": model_name,
            "Male": acc_male * 100,
            "Female": acc_female * 100,
            "Non-binary": acc_nb * 100,
            "No Label": acc_no_label * 100
        })
        
    df_acc = pd.DataFrame(acc_data)
    # Sort models by Model name
    df_acc = df_acc.sort_values("Model", ascending=True).reset_index(drop=True)
    
    # We will use pandas plotting for grouped bars
    df_acc_melted = pd.melt(df_acc, id_vars=["Model"], value_vars=["Male", "Female", "Non-binary", "No Label"],
                            var_name="Condition", value_name="Accuracy (%)")
                            
    plt.figure(figsize=(12, 8))
    
    sns.barplot(
        x="Accuracy (%)", 
        y="Model", 
        hue="Condition", 
        data=df_acc_melted,
        palette=["cornflowerblue", "lightcoral", "mediumseagreen", "gray"]
    )
    
    plt.title("Exact Match Accuracy by Sex-Label Condition\n(Sorted by Model name)", pad=15)
    plt.xlabel("Accuracy (%)")
    plt.ylabel("Model")
    plt.legend(title="Condition", loc='lower right')
    plt.grid(axis='x', linestyle='--', alpha=0.7, zorder=0)
    
    plt.xlim(0, 100)
    plt.tight_layout()
    
    out_path_acc = os.path.join(output_dir, "accuracy_by_condition_grouped_bar.png")
    plt.savefig(out_path_acc, dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Accuracy Delta Dot Plot (Male Anchor Effect) / Dumbbell Chart
    dumbbell_data = []
    for res in all_results:
        model_name = res["model"]
        ed = res.get("effect_decomposition", {})
        
        acc_male = ed.get("L3_accuracy_male", 0.0) * 100
        acc_no_label = ed.get("L1_accuracy_without_sex_info", 0.0) * 100
        delta = acc_male - acc_no_label
        
        dumbbell_data.append({
            "Model": model_name,
            "Male Accuracy": acc_male,
            "No Label Accuracy": acc_no_label,
            "Delta": delta
        })
        
    df_db = pd.DataFrame(dumbbell_data)
    # Sort by the delta (Male Anchor Effect) from highest positive to lowest negative, then by Model name
    df_db = df_db.sort_values(["Delta", "Model"], ascending=[True, True]).reset_index(drop=True)
    
    plt.figure(figsize=(10, 8))
    
    # Plot the lines connecting the dots
    plt.hlines(
        y=df_db["Model"], 
        xmin=df_db[["Male Accuracy", "No Label Accuracy"]].min(axis=1), 
        xmax=df_db[["Male Accuracy", "No Label Accuracy"]].max(axis=1), 
        color='gray', 
        alpha=0.5, 
        linewidth=2,
        zorder=1
    )
    
    # Plot the dots
    plt.scatter(df_db["No Label Accuracy"], df_db["Model"], color='gray', s=80, label='No Label (Baseline)', zorder=2)
    plt.scatter(df_db["Male Accuracy"], df_db["Model"], color='cornflowerblue', s=80, label='Male Condition', zorder=3)
    
    plt.title("Accuracy Delta: The Male Anchor Effect\n(Gap between Baseline and Male Condition)", pad=15)
    plt.xlabel("Accuracy (%)")
    plt.ylabel("Model")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='x', linestyle='--', alpha=0.7, zorder=0)
    
    plt.tight_layout()
    
    out_path_db = os.path.join(output_dir, "male_anchor_effect_dumbbell.png")
    plt.savefig(out_path_db, dpi=300, bbox_inches='tight')
    plt.close()

    # 6. ESI Score Distribution Grouped Bar Chart
    try:
        raw_df = pd.read_csv("eval/merged_evaluations.csv")
        
        # Ground truth
        gt_scores = raw_df["actual_score"].dropna().astype(int).tolist()
        
        dist_data = []
        
        # Add ground truth distribution for reference
        for score in gt_scores:
            dist_data.append({"Condition": "Ground Truth", "ESI Score": score})
            
        # Collect predictions across all models for key variants
        pred_cols = [c for c in raw_df.columns if c.startswith("predicted_score__")]
        
        for col in pred_cols:
            if col.endswith("__female"):
                cond = "Female"
            elif col.endswith("__male"):
                cond = "Male"
            elif col.endswith("__nb_ambiguous"):
                cond = "No Label"
            elif col.endswith("__nb_label_only"):
                cond = "Non-binary"
            else:
                continue
                
            scores = raw_df[col].dropna().astype(int).tolist()
            for score in scores:
                dist_data.append({"Condition": cond, "ESI Score": score})
                
        df_dist = pd.DataFrame(dist_data)
        
        # Calculate percentages
        dist_counts = df_dist.groupby(['Condition', 'ESI Score']).size().reset_index(name='Count')
        dist_totals = dist_counts.groupby('Condition')['Count'].transform('sum')
        dist_counts['Percentage'] = (dist_counts['Count'] / dist_totals) * 100
        
        plt.figure(figsize=(12, 6))
        
        conditions = ["Ground Truth", "No Label", "Male", "Female", "Non-binary"]
        colors = ["black", "gray", "cornflowerblue", "lightcoral", "mediumseagreen"]
        pal = dict(zip(conditions, colors))
        
        sns.barplot(
            data=dist_counts,
            x="ESI Score",
            y="Percentage",
            hue="Condition",
            hue_order=conditions,
            palette=pal
        )
        
        plt.title("ESI Score Distribution Across All Models by Condition\n(Percentage of cases assigned to each ESI level)", pad=15)
        plt.xlabel("ESI Score (1 = Most Urgent, 5 = Least Urgent)")
        plt.ylabel("Percentage of Assigned Scores (%)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title="Condition")
        
        out_path_dist = os.path.join(output_dir, "esi_score_distribution_grouped_bar.png")
        plt.savefig(out_path_dist, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Could not generate Distribution plot: {e}")

    # 7. Transition Sankey Diagram
    try:
        import plotly.graph_objects as go
        
        # Aggregate transition matrices across all models
        agg_matrices = {}
        for res in all_results:
            tm = res.get("transition_matrices", {})
            for variant, matrix in tm.items():
                if variant not in agg_matrices:
                    agg_matrices[variant] = matrix.copy()
                else:
                    agg_matrices[variant] += matrix
                    
        # Node setup
        labels = [f"Baseline ESI {i}" for i in range(1, 6)]
        colors = ["#333333"] * 5
        
        # Explicit coordinates to force vertical ordering.
        # Plotly coordinates are [0, 1]. We will space ESI 1..5 evenly vertically.
        node_x = [0.0] * 5
        # Slight vertical offsets so it looks nice
        base_y_positions = [0.1, 0.3, 0.5, 0.7, 0.9]
        node_y = base_y_positions.copy()
        
        # We interleave Male, Female, NB for each ESI level (1 to 5)
        for i in range(1, 6):
            labels.extend([f"Male ESI {i}", f"Female ESI {i}", f"NB ESI {i}"])
            colors.extend(["cornflowerblue", "lightcoral", "mediumseagreen"])
            node_x.extend([1.0] * 3)
            # Create a tight vertical cluster for the 3 target nodes around their base position
            by = base_y_positions[i-1]
            node_y.extend([by - 0.05, by, by + 0.05])
            
        sources = []
        targets = []
        values = []
        link_colors = []
        
        variant_offsets = {"male": 0, "female": 1, "nb_label_only": 2}
        
        for variant, offset in variant_offsets.items():
            if variant not in agg_matrices:
                continue
            matrix = agg_matrices[variant]
            for baseline_esi in range(1, 6):
                for variant_esi in range(1, 6):
                    try:
                        count = int(matrix.loc[baseline_esi, variant_esi])
                    except KeyError:
                        count = 0
                        
                    if count > 0:
                        sources.append(baseline_esi - 1)
                        # The nodes 0-4 are Baseline.
                        # Target nodes start at index 5. Each ESI level block takes 3 nodes.
                        # So target_idx = 5 + (variant_esi - 1)*3 + offset
                        target_idx = 5 + (variant_esi - 1)*3 + offset
                        targets.append(target_idx)
                        values.append(count)
                        
                        # Color logic
                        if baseline_esi == 2 and variant_esi == 3:
                            # Dangerous 2->3
                            link_colors.append("rgba(255, 0, 0, 0.6)")
                        elif baseline_esi == 2 and variant_esi == 5:
                            # Critical 2->5
                            link_colors.append("rgba(255, 0, 0, 0.8)")
                        elif baseline_esi == variant_esi:
                            # Stable
                            link_colors.append("rgba(200, 200, 200, 0.1)")
                        elif baseline_esi < variant_esi:
                            # Under-triage
                            link_colors.append("rgba(255, 165, 0, 0.4)")
                        else:
                            # Over-triage
                            link_colors.append("rgba(135, 206, 235, 0.4)")
                            
        fig = go.Figure(data=[go.Sankey(
            arrangement = "snap",
            node = dict(
              pad = 15,
              thickness = 20,
              line = dict(color = "black", width = 0.5),
              label = labels,
              color = colors,
              x = node_x,
              y = node_y
            ),
            link = dict(
              source = sources,
              target = targets,
              value = values,
              color = link_colors
            )
        )])
        
        fig.update_layout(
            title_text="Aggregated ESI Score Transitions (Baseline -> Labeled Conditions)<br>Red = Dangerous 2→3/5 Under-triage, Orange = Other Under-triage, Blue = Over-triage", 
            font_size=12
        )
        
        out_path_sankey = os.path.join(output_dir, "transition_sankey_diagram.png")
        fig.write_image(out_path_sankey, width=1200, height=800, scale=2)
        
    except ImportError:
        print("Plotly or kaleido not installed. Skipping Sankey diagram.")
    except Exception as e:
        print(f"Error generating Sankey diagram: {e}")

    # 8. Clinical Category Vulnerability Heatmap
    cat_data = []
    
    # Identify all unique categories present across all models
    all_categories = set()
    for res in all_results:
        vuln_cat = res.get("vulnerability_by_category")
        if isinstance(vuln_cat, pd.DataFrame) and not vuln_cat.empty:
            all_categories.update(vuln_cat["clinical_category"].unique())
            
    # Sort categories alphabetically for consistent columns, removing 'unclassified' if desired
    cat_columns = sorted(list(all_categories))
    if "unclassified" in cat_columns:
        cat_columns.remove("unclassified")
        cat_columns.append("unclassified") # move to end
        
    sorted_models = df["Model"].tolist() # use the PSS-sorted model list
    
    for model_name in sorted_models:
        res = next((r for r in all_results if r["model"] == model_name), None)
        if not res or "vulnerability_by_category" not in res:
            cat_data.append([0.0] * len(cat_columns))
            continue
            
        vuln_cat = res["vulnerability_by_category"]
        row_vals = []
        for cat in cat_columns:
            # We plot the pct_any_disagreement (fraction of cases that had any variant mismatch)
            if isinstance(vuln_cat, pd.DataFrame) and not vuln_cat.empty:
                cat_row = vuln_cat[vuln_cat["clinical_category"] == cat]
                if not cat_row.empty:
                    disagreement_rate = cat_row.iloc[0]["pct_any_disagreement"]
                else:
                    disagreement_rate = 0.0
            else:
                disagreement_rate = 0.0
                
            row_vals.append(disagreement_rate * 100) # convert to percentage
            
        cat_data.append(row_vals)
        
    df_cat_hm = pd.DataFrame(cat_data, index=sorted_models, columns=cat_columns)
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        df_cat_hm, 
        annot=True, 
        fmt=".1f", 
        cmap="Reds", 
        cbar_kws={'label': 'Disagreement Rate Across Variants (%)'},
        linewidths=.5
    )
    plt.title("Clinical Category Vulnerability by Model\n(Percentage of cases with any sex-based disagreement)", pad=15)
    plt.ylabel("Model (sorted by PSS)")
    plt.xlabel("Chief Complaint Category")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    out_path_cat_hm = os.path.join(output_dir, "clinical_category_vulnerability_heatmap.png")
    plt.savefig(out_path_cat_hm, dpi=300, bbox_inches='tight')
    plt.close()

    # 9. Per-Vignette Disagreement Strip Plot
    from util import categorize_complaint
    
    try:
        raw_df = pd.read_csv("eval/merged_evaluations.csv")
        # Map prompt hash to category and a concise label
        hash_to_cat = {}
        hash_to_text = {}
        for _, row in raw_df.iterrows():
            ph = row["prompt_hash"]
            prompt_text = str(row["prompt"])
            cat = categorize_complaint(prompt_text)
            hash_to_cat[ph] = cat
            hash_to_text[ph] = prompt_text[:30] + "..."
            
        vignette_data = []
        for res in all_results:
            model_name = res["model"]
            if "case_detail" in res:
                df_case = res["case_detail"]
                # case_detail is a DataFrame
                for _, row in df_case.iterrows():
                    ph = row["prompt_hash"]
                    pr = row["prediction_range"]
                    vignette_data.append({
                        "Model": model_name,
                        "Prompt Hash": ph,
                        "Prediction Range": pr,
                        "Category": hash_to_cat.get(ph, "Unknown"),
                        "Snippet": hash_to_text.get(ph, "Unknown")
                    })
                    
        df_vig = pd.DataFrame(vignette_data)
        
        # Sort vignettes by total/average prediction range across all models
        vig_sums = df_vig.groupby("Prompt Hash")["Prediction Range"].mean().sort_values(ascending=True)
        sorted_hashes = vig_sums.index.tolist()
        
        # To make the x-axis less cluttered, we might just map them to an index 1..70
        hash_to_idx = {h: i for i, h in enumerate(sorted_hashes)}
        df_vig["Vignette Rank"] = df_vig["Prompt Hash"].map(hash_to_idx)
        
        plt.figure(figsize=(16, 8))
        
        # Strip plot
        sns.stripplot(
            x="Vignette Rank", 
            y="Prediction Range", 
            hue="Category", 
            data=df_vig, 
            jitter=0.2, 
            alpha=0.6, 
            size=5
        )
        
        plt.title("Per-Vignette Disagreement (Max-Min ESI shift across variants)\nPoints are individual models. Sorted by mean vulnerability.", pad=15)
        plt.xlabel("Vignette (Sorted by aggregate sensitivity)")
        plt.ylabel("Prediction Range (ESI units)")
        
        # Only show a few x-ticks to avoid clutter, but show the grid
        plt.xticks([])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Move legend outside
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Clinical Category")
        plt.tight_layout()
        
        out_path_strip = os.path.join(output_dir, "per_vignette_disagreement_strip.png")
        plt.savefig(out_path_strip, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error generating Strip plot: {e}")

    # 10. Pairwise Condition Confusion Matrices (Small multiples)
    try:
        # We need the raw predictions to build arbitrary pairwise confusion matrices.
        # Let's aggregate them across all models.
        raw_df = pd.read_csv("eval/merged_evaluations.csv")
        
        # We want three specific pairwise comparisons:
        # 1. Female (Y) vs Male (X)
        # 2. Non-binary (Y) vs Male (X)
        # 3. Male (Y) vs No Label (X)
        
        matrix_f_vs_m = np.zeros((5, 5), dtype=int)
        matrix_nb_vs_m = np.zeros((5, 5), dtype=int)
        matrix_m_vs_nl = np.zeros((5, 5), dtype=int)
        
        for res in all_results:
            model_name = res["model"]
            
            # Find the column names for this model
            col_f = f"predicted_score__female__{model_name}"
            col_m = f"predicted_score__male__{model_name}"
            col_nb = f"predicted_score__nb_label_only__{model_name}"
            col_nl = f"predicted_score__nb_ambiguous__{model_name}"
            
            # Only proceed if all these columns exist
            if all(c in raw_df.columns for c in [col_f, col_m, col_nb, col_nl]):
                df_sub = raw_df[[col_f, col_m, col_nb, col_nl]].dropna().astype(int)
                
                for _, row in df_sub.iterrows():
                    # ESI is 1-5, matrix indices are 0-4
                    m_val = row[col_m] - 1
                    f_val = row[col_f] - 1
                    nb_val = row[col_nb] - 1
                    nl_val = row[col_nl] - 1
                    
                    if 0 <= m_val <= 4 and 0 <= f_val <= 4:
                        matrix_f_vs_m[f_val, m_val] += 1
                    if 0 <= m_val <= 4 and 0 <= nb_val <= 4:
                        matrix_nb_vs_m[nb_val, m_val] += 1
                    if 0 <= nl_val <= 4 and 0 <= m_val <= 4:
                        matrix_m_vs_nl[m_val, nl_val] += 1
                        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        def plot_cm(ax, matrix, title, xlabel, ylabel):
            sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5], cbar=False)
            ax.set_title(title, pad=15)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            
            # Highlight the ESI 2 -> 3 transition cell (X=2, Y=3 -> indices X=1, Y=2)
            # In matplotlib, rect coordinates are (x, y) starting from top left
            import matplotlib.patches as patches
            rect = patches.Rectangle((1, 2), 1, 1, fill=False, edgecolor='red', lw=3)
            ax.add_patch(rect)

        plot_cm(axes[0], matrix_f_vs_m, "Female vs. Male", "Male ESI", "Female ESI")
        plot_cm(axes[1], matrix_nb_vs_m, "Non-binary vs. Male", "Male ESI", "Non-binary ESI")
        plot_cm(axes[2], matrix_m_vs_nl, "Male vs. Baseline (No Label)", "Baseline (No Label) ESI", "Male ESI")
        
        plt.suptitle("Pairwise Condition Confusion Matrices (Aggregated Across All Models)\nRed box highlights the critical ESI 2 → 3 shift direction", y=1.05)
        plt.tight_layout()
        
        out_path_cm = os.path.join(output_dir, "pairwise_confusion_matrices.png")
        plt.savefig(out_path_cm, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error generating Confusion Matrices: {e}")

    # 11. Model Family × Condition Interaction Plot
    interaction_data = []
    
    for res in all_results:
        model_name = res["model"]
        family = get_model_family(model_name)
        ed = res.get("effect_decomposition", {})
        
        acc_no_label = ed.get("L1_accuracy_without_sex_info", 0.0) * 100
        acc_male = ed.get("L3_accuracy_male", 0.0) * 100
        acc_female = ed.get("L3_accuracy_female", 0.0) * 100
        acc_nb = ed.get("L4_accuracy_nonbinary", 0.0) * 100
        
        interaction_data.append({"Model": model_name, "Family": family, "Condition": "No Label", "Accuracy (%)": acc_no_label})
        interaction_data.append({"Model": model_name, "Family": family, "Condition": "Male", "Accuracy (%)": acc_male})
        interaction_data.append({"Model": model_name, "Family": family, "Condition": "Female", "Accuracy (%)": acc_female})
        interaction_data.append({"Model": model_name, "Family": family, "Condition": "Non-binary", "Accuracy (%)": acc_nb})
        
    df_int = pd.DataFrame(interaction_data)
    
    # Ensure correct order on the X-axis
    condition_order = ["No Label", "Male", "Female", "Non-binary"]
    
    plt.figure(figsize=(10, 6))
    
    sns.pointplot(
        data=df_int,
        x="Condition",
        y="Accuracy (%)",
        hue="Family",
        order=condition_order,
        dodge=0.2,
        errorbar="sd",
        capsize=0.05,
        palette="Set1",
        markers=["o", "s", "D", "^", "v", "p", "*"][:len(df_int["Family"].unique())]
    )
    
    plt.title("Model Family × Condition Interaction\n(Points show mean accuracy, error bars show standard deviation within family)", pad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Model Family", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    out_path_int = os.path.join(output_dir, "family_condition_interaction_plot.png")
    plt.savefig(out_path_int, dpi=300, bbox_inches='tight')
    plt.close()

    # 12. Mean ESI Deviation Diverging Bar Chart
    dev_data = []
    
    for model_name in sorted_models:
        res = next((r for r in all_results if r["model"] == model_name), None)
        if not res or "baseline_deviation" not in res:
            continue
            
        bd = res["baseline_deviation"]
        female_dev = bd.get("mean_deviation__female", 0.0)
        male_dev = bd.get("mean_deviation__male", 0.0)
        nb_dev = bd.get("mean_deviation__nb_label_only", 0.0)
        
        dev_data.append({"Model": model_name, "Condition": "Female", "Mean Deviation": female_dev})
        dev_data.append({"Model": model_name, "Condition": "Male", "Mean Deviation": male_dev})
        dev_data.append({"Model": model_name, "Condition": "Non-binary", "Mean Deviation": nb_dev})
        
    if dev_data:
        df_dev = pd.DataFrame(dev_data)
        
        plt.figure(figsize=(10, 10))
        sns.barplot(
            data=df_dev, 
            y="Model", 
            x="Mean Deviation", 
            hue="Condition",
            palette={"Female": "lightcoral", "Male": "cornflowerblue", "Non-binary": "mediumseagreen"}
        )
        
        # Add a vertical line at 0 for reference
        plt.axvline(0, color='black', linewidth=1.5, zorder=0)
        
        plt.title("Mean Signed ESI Deviation by Condition vs. Baseline\n(Negative = More Urgent / Upgraded, Positive = Less Urgent / Downgraded)", pad=15)
        plt.xlabel("Mean Deviation from Baseline (No Label) ESI")
        plt.ylabel("Model (sorted by PSS)")
        plt.grid(axis='x', linestyle='--', alpha=0.7, zorder=0)
        plt.legend(title="Condition", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        out_path_dev = os.path.join(output_dir, "mean_esi_deviation_diverging_bar.png")
        plt.savefig(out_path_dev, dpi=300, bbox_inches='tight')
        plt.close()

    # 13. Statistical Significance Volcano Plot
    import math
    
    volcano_data = []
    
    for res in all_results:
        model_name = res["model"]
        stats = res.get("statistical_significance", {})
        
        # We'll use Friedman's test p-value (since ESI is ordinal)
        p_val = stats.get("friedman_p")
        p_val_fdr = stats.get("friedman_fdr_p")
        
        # We need an effect size. Let's use the Perturbation Sensitivity Score (PSS)
        pss = res.get("sensitivity", {}).get("perturbation_sensitivity_score", 0.0)
        
        if p_val is not None and not math.isnan(p_val):
            # Avoid log(0) if p-value is extremely small
            p_val = max(p_val, 1e-15) 
            neg_log_p = -math.log10(p_val)
            
            volcano_data.append({
                "Model": model_name,
                "Family": get_model_family(model_name),
                "PSS (Effect Size)": pss,
                "-log10(p)": neg_log_p,
                "p_val": p_val,
                "p_val_fdr": p_val_fdr
            })
            
    if volcano_data:
        df_volc = pd.DataFrame(volcano_data)
        
        plt.figure(figsize=(10, 8))
        
        # Use scatterplot to color by family
        ax = sns.scatterplot(
            data=df_volc, 
            x="PSS (Effect Size)", 
            y="-log10(p)", 
            hue="Family", 
            s=100,
            palette="Set1",
            edgecolor="black"
        )
        
        # Add significance thresholds
        alpha = 0.05
        # Unadjusted threshold
        plt.axhline(-math.log10(alpha), color='red', linestyle='--', alpha=0.7, label=f'Unadjusted p=0.05')
        
        # We also want to show where FDR correction lands if it's available and makes sense.
        # For a simple representation, if any FDR p-value is < 0.05, we can show that limit.
        # Alternatively, we just mention it or plot the strict Bonferroni limit as a visual proxy.
        bonferroni_p = alpha / len(df_volc)
        plt.axhline(-math.log10(bonferroni_p), color='darkred', linestyle=':', alpha=0.7, label=f'Bonferroni p={bonferroni_p:.4f}')
        
        # Annotate models that are highly significant or have huge effect sizes
        for _, row in df_volc.iterrows():
            if row["p_val"] < 0.05 or row["PSS (Effect Size)"] > 0.15:
                # Add text label slightly offset
                ax.text(
                    row["PSS (Effect Size)"] + 0.005, 
                    row["-log10(p)"], 
                    row["Model"].split("_")[-1],  # Just show the short name
                    fontsize=9
                )

        plt.title("Statistical Significance vs. Effect Size (Volcano Plot)\n(Friedman Test for Ordinal ESI Shifts)", pad=15)
        plt.xlabel("Perturbation Sensitivity Score (Effect Size)")
        plt.ylabel("-log10(p-value)")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        out_path_volc = os.path.join(output_dir, "statistical_significance_volcano.png")
        plt.savefig(out_path_volc, dpi=300, bbox_inches='tight')
        plt.close()

    # 14. Counterfactual Vignette Example Panels (5 Cases)
    try:
        raw_df = pd.read_csv("eval/merged_evaluations.csv")
        import matplotlib.patches as patches
        
        target_hashes = []
        
        # 1. We know from analysis that GPT-5.4-nano has a 2->5 failure. Let's find it.
        nano_col_f = "predicted_score__female__openai_gpt-5.4-nano-2026-03-17"
        nano_col_bl = "predicted_score__nb_ambiguous__openai_gpt-5.4-nano-2026-03-17"
        
        if nano_col_f in raw_df.columns and nano_col_bl in raw_df.columns:
            fail_cases = raw_df[(raw_df[nano_col_bl] == 2) & ((raw_df[nano_col_f] == 5) | (raw_df[nano_col_f] == 4))]
            if not fail_cases.empty:
                fail_cases = fail_cases.copy()
                fail_cases.loc[:, "diff"] = abs(fail_cases[nano_col_bl] - fail_cases[nano_col_f])
                target_hashes.append(fail_cases.sort_values("diff", ascending=False).iloc[0]["prompt_hash"])
                
        # 2. Find the vignettes with the highest average prediction range across all models
        vig_ranges = {}
        for res in all_results:
            if "case_detail" in res:
                df_case = res["case_detail"]
                for _, r in df_case.iterrows():
                    ph = r["prompt_hash"]
                    pr = r["prediction_range"]
                    vig_ranges[ph] = vig_ranges.get(ph, 0) + pr
                    
        sorted_vig = sorted(vig_ranges.keys(), key=lambda k: vig_ranges[k], reverse=True)
        
        for h in sorted_vig:
            if h not in target_hashes:
                target_hashes.append(h)
            if len(target_hashes) >= 15:
                break
                
        # We will pick 3 representative models
        sorted_models = df["Model"].tolist()
        best_model = sorted_models[0]
        med_model = sorted_models[len(sorted_models)//2 + 1]
        worst_model = sorted_models[-1]
        
        rep_models = [
            ("Robust (Low Sensitivity)", best_model),
            ("Average (Med Sensitivity)", med_model),
            ("Brittle (High Sensitivity)", worst_model)
        ]
        
        for idx, target_hash in enumerate(target_hashes, 1):
            row = raw_df[raw_df["prompt_hash"] == target_hash].iloc[0]
            prompt_text = str(row["prompt"])
            
            # Create a figure for the text panel
            fig = plt.figure(figsize=(14, 8))
            fig.patch.set_facecolor('#f8f9fa')
            
            plt.axis('off')
            
            # Title
            plt.text(0.5, 0.95, f"Counterfactual Vignette Analysis: The Demographic Shift Effect", 
                     ha='center', va='top', fontsize=18, fontweight='bold', fontfamily='sans-serif')
            plt.text(0.5, 0.90, "Identical vitals and chief complaint. Only the demographic token changes.", 
                     ha='center', va='top', fontsize=12, style='italic', color='#555555')
            
            # Parse prompt to make it look nice
            lines = prompt_text.split('\n')
            cc = lines[0] if len(lines) > 0 else "Chief complaint: Unknown"
            vitals = lines[1] if len(lines) > 1 else ""
            
            plt.text(0.12, 0.81, "[SEX: ____]", fontsize=12, fontweight='bold', color='#d9534f', family='monospace', zorder=2)
            plt.text(0.12, 0.74, cc, fontsize=14, fontweight='bold', zorder=2)
            plt.text(0.12, 0.68, vitals, fontsize=11, family='monospace', zorder=2)
            
            # Draw a table of predictions
            plt.text(0.5, 0.58, "Assigned ESI Score (1 = Resuscitation, 5 = Non-urgent)", 
                     ha='center', va='top', fontsize=14, fontweight='bold')
                     
            # Table headers
            conditions = [
                ("Baseline (Omitted)", "nb_ambiguous", "gray"),
                ("Female", "female", "lightcoral"),
                ("Male", "male", "cornflowerblue"),
                ("Non-binary", "nb_label_only", "mediumseagreen")
            ]
            
            # X positions
            x_start = 0.35
            x_step = 0.15
            
            for i, (cond_name, _, color) in enumerate(conditions):
                plt.text(x_start + i*x_step, 0.50, cond_name, ha='center', fontweight='bold', color=color, fontsize=12)
                
            y_start = 0.40
            y_step = 0.12
            
            for i, (label, m_name) in enumerate(rep_models):
                y_pos = y_start - i*y_step
                # Draw row background
                rect = patches.Rectangle((0.05, y_pos - 0.05), 0.9, 0.1, linewidth=0, facecolor='#eeeeee' if i%2==0 else 'white', zorder=0)
                ax.add_patch(rect)
                
                # Model name
                clean_name = m_name.split("_")[-1]
                plt.text(0.08, y_pos, f"{label}\n{clean_name}", va='center', fontweight='bold', fontsize=11)
                
                # Scores
                for j, (_, suffix, color) in enumerate(conditions):
                    col_name = f"predicted_score__{suffix}__{m_name}"
                    val = "N/A"
                    if col_name in row:
                        val = str(int(row[col_name]))
                        
                    plt.text(x_start + j*x_step, y_pos, val, ha='center', va='center', fontsize=18, fontweight='bold', color=color)
                    
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            
            out_path_panel = os.path.join(output_dir, f"counterfactual_vignette_panel_case_{idx}.png")
            plt.savefig(out_path_panel, dpi=300, bbox_inches='tight')
            plt.close()
            
    except Exception as e:
        print(f"Error generating Counterfactual Vignette Panel: {e}")