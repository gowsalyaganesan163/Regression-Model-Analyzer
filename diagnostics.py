import os
import matplotlib.pyplot as plt
import numpy as np


def overfitting_analysis(results):
    """
    Compares train and test R2 scores.
    """

    diagnostics = {}

    for model_name, result in results.items():

        train_r2 = result["train_r2"]
        test_r2 = result["test_r2"]

        gap = train_r2 - test_r2

        if gap >= 0.15:

            status = (
                "Possible overfitting: "
                "training performance is considerably "
                "higher than test performance."
            )

        elif train_r2 < 0.60 and test_r2 < 0.60:

            status = (
                "Possible underfitting or weak predictive "
                "relationship."
            )

        else:

            status = (
                "No large train-test R2 gap detected."
            )

        diagnostics[model_name] = {
            "train_r2": train_r2,
            "test_r2": test_r2,
            "gap": gap,
            "status": status
        }

    return diagnostics


def residual_analysis(
        results,
        output_dir
):
    """
    Creates residual plots for each model.
    """

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    residual_paths = {}

    for model_name, result in results.items():

        y_actual = np.asarray(
            result["y_test"]
        )

        predictions = np.asarray(
            result["predictions"]
        )

        residuals = (
            y_actual - predictions
        )

        safe_name = (
            model_name
            .replace(" ", "_")
            .replace("/", "_")
        )

        path = os.path.join(
            output_dir,
            f"residual_{safe_name}.png"
        )

        plt.figure(figsize=(8, 5))

        plt.scatter(
            predictions,
            residuals,
            alpha=0.7
        )

        plt.axhline(
            y=0,
            linestyle="--"
        )

        plt.xlabel(
            "Predicted Values"
        )

        plt.ylabel(
            "Residuals"
        )

        plt.title(
            f"Residual Analysis - {model_name}"
        )

        plt.tight_layout()

        plt.savefig(
            path,
            dpi=150
        )

        plt.close()

        residual_paths[model_name] = path

    return residual_paths