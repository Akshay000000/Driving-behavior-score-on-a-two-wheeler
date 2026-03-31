from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_top_frequent_itemsets(
    frequent_itemsets: pd.DataFrame,
    top_n: int = 15,
    figsize: tuple[int, int] = (8, 5),
) -> Optional[plt.Figure]:
    """Create a bar plot of top-N frequent itemsets by support."""
    if frequent_itemsets.empty:
        return None

    top = frequent_itemsets.head(top_n).copy()
    top["itemsets_str"] = top["itemsets"].apply(lambda s: ", ".join(sorted(list(s))))

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=top,
        x="support",
        y="itemsets_str",
        hue="itemsets_str",
        ax=ax,
        palette="viridis",
        legend=False,
    )
    ax.set_xlabel("Support")
    ax.set_ylabel("Itemset (Symptoms)")
    ax.set_title(f"Top {min(top_n, len(top))} Frequent Symptom Itemsets")
    plt.tight_layout()
    return fig


def plot_cooccurrence_heatmap(
    co_matrix: pd.DataFrame,
    max_symptoms: int = 25,
    figsize: tuple[int, int] = (8, 6),
) -> Optional[plt.Figure]:
    """Create a heatmap of symptom co-occurrence counts."""
    if co_matrix.empty:
        return None

    # Limit to top symptoms by total co-occurrence
    totals = co_matrix.sum(axis=1).sort_values(ascending=False)
    top_indices = totals.head(max_symptoms).index
    sub_matrix = co_matrix.loc[top_indices, top_indices]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        sub_matrix,
        cmap="Reds",
        linewidths=0.5,
        annot=False,
        ax=ax,
    )
    ax.set_title("Symptom Co-occurrence Heatmap (Counts)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig
