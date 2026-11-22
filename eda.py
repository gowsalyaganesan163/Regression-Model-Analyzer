import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def automatic_eda(df, target, output_dir):
    """
    Performs automatic EDA and saves a target distribution plot.
    """

    os.makedirs(output_dir, exist_ok=True)

    numeric_df = df.select_dtypes(include=np.number)

    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_columns": len(
            df.select_dtypes(include=np.number).columns
        ),
        "categorical_columns": len(
            df.select_dtypes(exclude=np.number).columns
        ),
        "missing_values": int(
            df.isnull().sum().sum()
        ),
        "duplicate_rows": int(
            df.duplicated().sum()
        )
    }

    # Target distribution

    target_plot = os.path.join(
        output_dir,
        "target_distribution.png"
    )

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df[target],
        kde=True
    )

    plt.title(f"Distribution of {target}")
    plt.xlabel(target)
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(target_plot, dpi=150)
    plt.close()

    return summary, target_plot


def correlation_analysis(df, target, output_dir):
    """
    Creates a correlation heatmap.
    """

    os.makedirs(output_dir, exist_ok=True)

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.shape[1] < 2:
        return None, []

    correlation = numeric_df.corr()

    plot_path = os.path.join(
        output_dir,
        "correlation_heatmap.png"
    )

    plt.figure(
        figsize=(
            max(8, len(correlation.columns) * 0.7),
            max(6, len(correlation.columns) * 0.6)
        )
    )

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    correlations_with_target = (
        correlation[target]
        .drop(target)
        .sort_values(
            key=lambda x: x.abs(),
            ascending=False
        )
    )

    return plot_path, correlations_with_target


def scatter_analysis(df, target, output_dir):
    """
    Creates scatter plots between target and
    important numerical predictors.
    """

    os.makedirs(output_dir, exist_ok=True)

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    numeric_columns.remove(target)

    if not numeric_columns:
        return []

    correlations = (
        df[numeric_columns + [target]]
        .corr()[target]
        .drop(target)
        .abs()
        .sort_values(ascending=False)
    )

    selected_columns = correlations.head(6).index.tolist()

    plot_paths = []

    for column in selected_columns:

        path = os.path.join(
            output_dir,
            f"scatter_{column}.png"
        )

        plt.figure(figsize=(7, 5))

        plt.scatter(
            df[column],
            df[target],
            alpha=0.7
        )

        plt.xlabel(column)
        plt.ylabel(target)

        plt.title(
            f"{column} vs {target}"
        )

        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

        plot_paths.append(path)

    return plot_paths


def pair_plot_analysis(df, target, output_dir):
    """
    Creates a pair plot using target and the
    most relevant numerical features.
    """

    os.makedirs(output_dir, exist_ok=True)

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    numeric_predictors = [
        col for col in numeric_columns
        if col != target
    ]

    if not numeric_predictors:
        return None

    correlations = (
        df[numeric_predictors + [target]]
        .corr()[target]
        .drop(target)
        .abs()
        .sort_values(ascending=False)
    )

    selected = correlations.head(4).index.tolist()

    columns = selected + [target]

    pair_df = df[columns].copy()

    # Avoid very large pair plots
    if len(pair_df) > 500:
        pair_df = pair_df.sample(
            500,
            random_state=42
        )

    grid = sns.pairplot(
        pair_df
    )

    grid.fig.suptitle(
        "Pair Plot Analysis",
        y=1.02
    )

    path = os.path.join(
        output_dir,
        "pair_plot.png"
    )

    grid.savefig(
        path,
        dpi=150
    )

    plt.close("all")

    return path