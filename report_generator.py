import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.styles import ParagraphStyle

from reportlab.lib.units import inch

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak
)


def add_image(
        story,
        image_path,
        width=6.3 * inch
):

    if image_path and os.path.exists(image_path):

        image = Image(
            image_path,
            width=width,
            height=width * 0.62
        )

        story.append(image)
        story.append(
            Spacer(1, 12)
        )


def generate_report(
        output_path,
        dataset_name,
        target,
        validation,
        cleaning_log,
        eda_summary,
        target_plot,
        correlation_plot,
        scatter_plots,
        pair_plot,
        comparison_df,
        overfitting_results,
        residual_plots
):

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=22,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading2"],
        fontSize=15,
        spaceBefore=15,
        spaceAfter=8
    )

    normal_style = styles["BodyText"]

    story = []

    # =================================================
    # TITLE
    # =================================================

    story.append(
        Paragraph(
            "REGRESSION MODEL ANALYSIS REPORT",
            title_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Dataset:</b> {dataset_name}",
            normal_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Target Variable:</b> {target}",
            normal_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Generated:</b> "
            f"{datetime.now().strftime('%d-%m-%Y %H:%M')}",
            normal_style
        )
    )

    story.append(
        Spacer(1, 20)
    )

    # =================================================
    # EXECUTIVE SUMMARY
    # =================================================

    story.append(
        Paragraph(
            "1. Executive Summary",
            heading_style
        )
    )

    if not comparison_df.empty:

        best_model = comparison_df.iloc[0]["Model"]
        best_score = comparison_df.iloc[0]["Test R2"]

        summary_text = (
            f"The analysis evaluated "
            f"{len(comparison_df)} regression model(s) "
            f"for predicting <b>{target}</b>. "
            f"The highest test R² among the evaluated models "
            f"was achieved by <b>{best_model}</b>, "
            f"with a test R² of <b>{best_score:.4f}</b>. "
            f"Model performance should be interpreted in the "
            f"context of the dataset, preprocessing steps and "
            f"train-test split used in this analysis."
        )

    else:

        summary_text = (
            "No model comparison results were available."
        )

    story.append(
        Paragraph(
            summary_text,
            normal_style
        )
    )

    # =================================================
    # DATASET QUALITY
    # =================================================

    story.append(
        Paragraph(
            "2. Dataset & Data Quality",
            heading_style
        )
    )

    quality_data = [
        ["Metric", "Value"],
        ["Rows", validation["rows"]],
        ["Columns", validation["columns"]],
        ["Missing Cells", validation["missing_cells"]],
        ["Duplicate Rows", validation["duplicates"]],
        ["Missing Target Values", validation["target_missing"]]
    ]

    table = Table(
        quality_data,
        colWidths=[3 * inch, 2 * inch]
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "ALIGN",
                    (1, 1),
                    (-1, -1),
                    "CENTER"
                )
            ]
        )
    )

    story.append(table)

    story.append(
        Spacer(1, 12)
    )

    story.append(
        Paragraph(
            "<b>Automatic Cleaning Actions:</b>",
            normal_style
        )
    )

    if cleaning_log:

        for action in cleaning_log:

            story.append(
                Paragraph(
                    "• " + action,
                    normal_style
                )
            )

    else:

        story.append(
            Paragraph(
                "• No cleaning actions were required.",
                normal_style
            )
        )

    # =================================================
    # EDA
    # =================================================

    story.append(
        Paragraph(
            "3. Exploratory Data Analysis",
            heading_style
        )
    )

    story.append(
        Paragraph(
            f"The dataset contains "
            f"<b>{eda_summary['numeric_columns']}</b> "
            f"numeric feature(s) and "
            f"<b>{eda_summary['categorical_columns']}</b> "
            f"categorical feature(s).",
            normal_style
        )
    )

    add_image(
        story,
        target_plot
    )

    if correlation_plot:

        story.append(
            Paragraph(
                "<b>Correlation Analysis</b>",
                normal_style
            )
        )

        add_image(
            story,
            correlation_plot
        )

    if scatter_plots:

        story.append(
            Paragraph(
                "<b>Scatter Plot Analysis</b>",
                normal_style
            )
        )

        for plot in scatter_plots:

            add_image(
                story,
                plot
            )

    if pair_plot:

        story.append(
            Paragraph(
                "<b>Pair Plot Analysis</b>",
                normal_style
            )
        )

        add_image(
            story,
            pair_plot
        )

    # =================================================
    # MODEL COMPARISON
    # =================================================

    story.append(PageBreak())

    story.append(
        Paragraph(
            "4. Regression Model Comparison",
            heading_style
        )
    )

    comparison_data = [
        [
            "Model",
            "Train R²",
            "Test R²",
            "R² Gap"
        ]
    ]

    for _, row in comparison_df.iterrows():

        comparison_data.append(
            [
                row["Model"],
                f"{row['Train R2']:.4f}",
                f"{row['Test R2']:.4f}",
                f"{row['R2 Gap']:.4f}"
            ]
        )

    comparison_table = Table(
        comparison_data,
        colWidths=[
            2.5 * inch,
            1 * inch,
            1 * inch,
            1 * inch
        ]
    )

    comparison_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "ALIGN",
                    (1, 1),
                    (-1, -1),
                    "CENTER"
                )
            ]
        )
    )

    story.append(
        comparison_table
    )

    # =================================================
    # OVERFITTING
    # =================================================

    if overfitting_results:

        story.append(
            Paragraph(
                "5. Overfitting Analysis",
                heading_style
            )
        )

        for model_name, result in (
            overfitting_results.items()
        ):

            story.append(
                Paragraph(
                    f"<b>{model_name}</b>",
                    normal_style
                )
            )

            story.append(
                Paragraph(
                    f"Train R²: "
                    f"{result['train_r2']:.4f} | "
                    f"Test R²: "
                    f"{result['test_r2']:.4f} | "
                    f"Gap: "
                    f"{result['gap']:.4f}",
                    normal_style
                )
            )

            story.append(
                Paragraph(
                    result["status"],
                    normal_style
                )
            )

            story.append(
                Spacer(1, 8)
            )

    # =================================================
    # RESIDUAL ANALYSIS
    # =================================================

    if residual_plots:

        story.append(
            PageBreak()
        )

        story.append(
            Paragraph(
                "6. Residual Analysis",
                heading_style
            )
        )

        story.append(
            Paragraph(
                "Residuals represent the difference between "
                "the actual target values and the model's "
                "predicted values. The plots below help "
                "identify systematic prediction errors.",
                normal_style
            )
        )

        for model_name, plot in (
            residual_plots.items()
        ):

            story.append(
                Paragraph(
                    f"<b>{model_name}</b>",
                    normal_style
                )

            )

            add_image(
                story,
                plot
            )

    # =================================================
    # METHODOLOGY
    # =================================================

    story.append(
        Paragraph(
            "7. Methodology",
            heading_style
        )
    )

    methodology = [
        "1. User selects a CSV dataset.",
        "2. Dataset validation is performed automatically.",
        "3. Missing values, duplicates and unsuitable columns are handled automatically.",
        "4. Exploratory Data Analysis is performed.",
        "5. User selects relationship analyses and regression techniques.",
        "6. Selected regression models are trained using an 80/20 train-test split.",
        "7. R² is used to evaluate regression performance.",
        "8. Optional overfitting and residual diagnostics are performed.",
        "9. This technical report is generated automatically."
    ]

    for item in methodology:

        story.append(
            Paragraph(
                item,
                normal_style
            )
        )

    # =================================================
    # FINAL ASSESSMENT
    # =================================================

    story.append(
        Paragraph(
            "8. Final Model Assessment",
            heading_style
        )
    )

    if not comparison_df.empty:

        best_model = comparison_df.iloc[0]["Model"]
        best_score = comparison_df.iloc[0]["Test R2"]

        story.append(
            Paragraph(
                f"Among the models evaluated in this run, "
                f"<b>{best_model}</b> produced the highest "
                f"test R² of <b>{best_score:.4f}</b>. "
                f"This indicates the strongest test-set "
                f"fit among the models included in this "
                f"particular analysis.",
                normal_style
            )
        )

        story.append(
            Spacer(1, 10)
        )

        story.append(
            Paragraph(
                "This result should not automatically be "
                "treated as proof that the model will perform "
                "equally well on unseen real-world data. "
                "Additional validation and domain-specific "
                "testing may be required before deployment.",
                normal_style
            )
        )

    doc.build(story)

    return output_path