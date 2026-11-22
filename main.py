import os
import tkinter as tk

from tkinter import ttk
from tkinter import messagebox

from data_loader import (
    select_csv_file,
    load_csv
)

from data_cleaning import (
    validate_dataset,
    clean_dataset
)

from eda import (
    automatic_eda,
    correlation_analysis,
    scatter_analysis,
    pair_plot_analysis
)

from models import (
    train_selected_models
)

from diagnostics import (
    overfitting_analysis,
    residual_analysis
)

from report_generator import (
    generate_report
)


class RegressionAnalyzer:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Regression Model Analyzer"
        )

        # Increased window height
        self.root.geometry(
            "900x900"
        )

        self.root.minsize(
            850,
            850
        )

        self.df = None
        self.file_path = None

        self.create_gui()

    # =================================================
    # CREATE GUI
    # =================================================

    def create_gui(self):

        # -------------------------------------------------
        # Title
        # -------------------------------------------------

        title = tk.Label(
            self.root,
            text="REGRESSION MODEL ANALYZER",
            font=("Arial", 22, "bold")
        )

        title.pack(
            pady=(20, 5)
        )

        subtitle = tk.Label(
            self.root,
            text=(
                "Automated Data Validation • Cleaning • EDA "
                "• Regression • Diagnostics • PDF Report"
            ),
            font=("Arial", 10)
        )

        subtitle.pack(
            pady=(0, 15)
        )

        # =================================================
        # 1. DATASET
        # =================================================

        dataset_frame = ttk.LabelFrame(
            self.root,
            text="1. Dataset"
        )

        dataset_frame.pack(
            fill="x",
            padx=30,
            pady=8
        )

        self.file_label = tk.Label(
            dataset_frame,
            text="No dataset selected",
            anchor="w"
        )

        self.file_label.pack(
            side="left",
            padx=10,
            pady=10,
            fill="x",
            expand=True
        )

        browse_button = ttk.Button(
            dataset_frame,
            text="SELECT CSV",
            command=self.browse_dataset
        )

        browse_button.pack(
            side="right",
            padx=10,
            pady=10
        )

        # =================================================
        # 2. TARGET VARIABLE
        # =================================================

        target_frame = ttk.LabelFrame(
            self.root,
            text="2. Target Variable"
        )

        target_frame.pack(
            fill="x",
            padx=30,
            pady=8
        )

        tk.Label(
            target_frame,
            text="Target:"
        ).pack(
            side="left",
            padx=10,
            pady=10
        )

        self.target_combo = ttk.Combobox(
            target_frame,
            state="readonly",
            width=40
        )

        self.target_combo.pack(
            side="left",
            padx=10,
            pady=10
        )

        # =================================================
        # 3. AUTOMATIC PROCESSING
        # =================================================

        auto_frame = ttk.LabelFrame(
            self.root,
            text="3. Automatic Processing"
        )

        auto_frame.pack(
            fill="x",
            padx=30,
            pady=8
        )

        automatic_text = (
            "✓ Dataset Validation\n"
            "✓ Data Cleaning\n"
            "✓ Exploratory Data Analysis"
        )

        tk.Label(
            auto_frame,
            text=automatic_text,
            justify="left",
            font=("Arial", 10)
        ).pack(
            anchor="w",
            padx=15,
            pady=8
        )

        # =================================================
        # 4. RELATIONSHIP ANALYSIS
        # =================================================

        relationship_frame = ttk.LabelFrame(
            self.root,
            text="4. Relationship Analysis"
        )

        relationship_frame.pack(
            fill="x",
            padx=30,
            pady=8
        )

        self.correlation_var = tk.BooleanVar(
            value=True
        )

        self.scatter_var = tk.BooleanVar(
            value=True
        )

        self.pairplot_var = tk.BooleanVar(
            value=False
        )

        ttk.Checkbutton(
            relationship_frame,
            text="Correlation Analysis",
            variable=self.correlation_var
        ).pack(
            side="left",
            padx=15,
            pady=8
        )

        ttk.Checkbutton(
            relationship_frame,
            text="Scatter Plot Analysis",
            variable=self.scatter_var
        ).pack(
            side="left",
            padx=15,
            pady=8
        )

        ttk.Checkbutton(
            relationship_frame,
            text="Pair Plot Analysis",
            variable=self.pairplot_var
        ).pack(
            side="left",
            padx=15,
            pady=8
        )

        # =================================================
        # 5. REGRESSION TECHNIQUES
        # =================================================

        model_frame = ttk.LabelFrame(
            self.root,
            text="5. Regression Techniques"
        )


        model_frame.pack(
            fill="x",
            expand=False,
            padx=30,
            pady=8
        )

        self.model_vars = {}

        model_names = [
            "Simple Linear Regression",
            "Multiple Linear Regression",
            "Polynomial Regression",
            "Ridge Regression",
            "Lasso Regression",
            "Decision Tree Regression",
            "Random Forest Regression"
        ]

        for i, model_name in enumerate(
            model_names
        ):

            var = tk.BooleanVar(
                value=True
            )

            self.model_vars[
                model_name
            ] = var

            ttk.Checkbutton(
                model_frame,
                text=model_name,
                variable=var
            ).grid(
                row=i // 2,
                column=i % 2,
                sticky="w",
                padx=20,
                pady=4
            )

        # =================================================
        # 6. MODEL EVALUATION
        # =================================================

        evaluation_frame = ttk.LabelFrame(
            self.root,
            text="6. Model Evaluation"
        )

        evaluation_frame.pack(
            fill="x",
            padx=30,
            pady=8
        )

        self.r2_var = tk.BooleanVar(
            value=True
        )

        ttk.Checkbutton(
            evaluation_frame,
            text="R² Score",
            variable=self.r2_var
        ).pack(
            side="left",
            padx=15,
            pady=8
        )

        # =================================================
        # 7. MODEL DIAGNOSTICS
        # =================================================

        diagnostic_frame = ttk.LabelFrame(
            self.root,
            text="7. Model Diagnostics"
        )

        diagnostic_frame.pack(
            fill="x",
            padx=30,
            pady=8
        )

        self.overfit_var = tk.BooleanVar(
            value=True
        )

        self.residual_var = tk.BooleanVar(
            value=True
        )

        ttk.Checkbutton(
            diagnostic_frame,
            text="Overfitting Analysis",
            variable=self.overfit_var
        ).pack(
            side="left",
            padx=15,
            pady=8
        )

        ttk.Checkbutton(
            diagnostic_frame,
            text="Residual Analysis",
            variable=self.residual_var
        ).pack(
            side="left",
            padx=15,
            pady=8
        )

        # =================================================
        # START ANALYSIS BUTTON
        # =================================================

        self.start_button = ttk.Button(
            self.root,
            text="START ANALYSIS",
            command=self.start_analysis
        )

        self.start_button.pack(
            pady=(15, 8)
        )

        # =================================================
        # STATUS
        # =================================================

        self.status_label = tk.Label(
            self.root,
            text="Ready",
            font=("Arial", 10)
        )

        self.status_label.pack(
            pady=(0, 15)
        )

    # =================================================
    # SELECT DATASET
    # =================================================

    def browse_dataset(self):

        file_path = select_csv_file()

        if not file_path:
            return

        try:

            df = load_csv(
                file_path
            )

            self.df = df
            self.file_path = file_path

            self.file_label.config(
                text=os.path.basename(
                    file_path
                )
            )

            # -------------------------------------------------
            # Prefer numerical columns as regression targets
            # -------------------------------------------------

            numeric_columns = (
                df.select_dtypes(
                    include="number"
                ).columns.tolist()
            )

            if numeric_columns:

                self.target_combo["values"] = (
                    numeric_columns
                )

                self.target_combo.set(
                    numeric_columns[-1]
                )

            else:

                self.target_combo["values"] = (
                    df.columns.tolist()
                )

                self.target_combo.set(
                    df.columns[-1]
                )

            self.status_label.config(
                text=(
                    f"Dataset loaded: "
                    f"{len(df)} rows × "
                    f"{len(df.columns)} columns"
                )
            )

        except Exception as e:

            messagebox.showerror(
                "Dataset Error",
                str(e)
            )

    # =================================================
    # START ANALYSIS
    # =================================================

    def start_analysis(self):

        # -------------------------------------------------
        # Check dataset
        # -------------------------------------------------

        if self.df is None:

            messagebox.showwarning(
                "Dataset Required",
                "Please select a CSV dataset first."
            )

            return

        # -------------------------------------------------
        # Check target
        # -------------------------------------------------

        target = (
            self.target_combo.get()
        )

        if not target:

            messagebox.showwarning(
                "Target Required",
                "Please select a target variable."
            )

            return

        # -------------------------------------------------
        # R² is required
        # -------------------------------------------------

        if not self.r2_var.get():

            messagebox.showwarning(
                "R² Required",
                "R² Score is required for regression "
                "model comparison."
            )

            return

        # -------------------------------------------------
        # Get selected models
        # -------------------------------------------------

        selected_models = [
            name
            for name, variable
            in self.model_vars.items()
            if variable.get()
        ]

        if not selected_models:

            messagebox.showwarning(
                "Model Required",
                "Please select at least one "
                "regression model."
            )

            return

        try:

            # Disable button during analysis

            self.start_button.config(
                state="disabled"
            )

            self.status_label.config(
                text="Starting analysis..."
            )

            self.root.update_idletasks()

            # =================================================
            # OUTPUT DIRECTORY
            # =================================================

            base_dir = os.path.dirname(
                self.file_path
            )

            dataset_name = os.path.splitext(
                os.path.basename(
                    self.file_path
                )
            )[0]

            output_dir = os.path.join(
                base_dir,
                f"{dataset_name}_Regression_Analysis"
            )

            os.makedirs(
                output_dir,
                exist_ok=True
            )

            eda_dir = os.path.join(
                output_dir,
                "EDA"
            )

            diagnostic_dir = os.path.join(
                output_dir,
                "Diagnostics"
            )

            os.makedirs(
                eda_dir,
                exist_ok=True
            )

            os.makedirs(
                diagnostic_dir,
                exist_ok=True
            )

            # =================================================
            # 1. AUTOMATIC VALIDATION
            # =================================================

            self.status_label.config(
                text="1/7 Validating dataset..."
            )

            self.root.update_idletasks()

            validation = validate_dataset(
                self.df,
                target
            )

            # =================================================
            # 2. AUTOMATIC DATA CLEANING
            # =================================================

            self.status_label.config(
                text="2/7 Cleaning dataset..."
            )

            self.root.update_idletasks()

            cleaned_df, cleaning_log = (
                clean_dataset(
                    self.df,
                    target
                )
            )

            # =================================================
            # 3. AUTOMATIC EDA
            # =================================================

            self.status_label.config(
                text="3/7 Performing EDA..."
            )

            self.root.update_idletasks()

            eda_summary, target_plot = (
                automatic_eda(
                    cleaned_df,
                    target,
                    eda_dir
                )
            )

            correlation_plot = None
            scatter_plots = []
            pair_plot = None

            # =================================================
            # 4. RELATIONSHIP ANALYSIS
            # =================================================

            self.status_label.config(
                text="4/7 Performing relationship analysis..."
            )

            self.root.update_idletasks()

            if self.correlation_var.get():

                correlation_plot, _ = (
                    correlation_analysis(
                        cleaned_df,
                        target,
                        eda_dir
                    )
                )

            if self.scatter_var.get():

                scatter_plots = (
                    scatter_analysis(
                        cleaned_df,
                        target,
                        eda_dir
                    )
                )

            if self.pairplot_var.get():

                pair_plot = (
                    pair_plot_analysis(
                        cleaned_df,
                        target,
                        eda_dir
                    )
                )

            # =================================================
            # 5. TRAIN REGRESSION MODELS
            # =================================================

            self.status_label.config(
                text="5/7 Training regression models..."
            )

            self.root.update_idletasks()

            results, comparison_df, comparison_path = (
                train_selected_models(
                    cleaned_df,
                    target,
                    selected_models,
                    output_dir
                )
            )

            # =================================================
            # 6. MODEL DIAGNOSTICS
            # =================================================

            overfitting_results = {}
            residual_plots = {}

            self.status_label.config(
                text="6/7 Performing diagnostics..."
            )

            self.root.update_idletasks()

            if self.overfit_var.get():

                overfitting_results = (
                    overfitting_analysis(
                        results
                    )
                )

            if self.residual_var.get():

                residual_plots = (
                    residual_analysis(
                        results,
                        diagnostic_dir
                    )
                )

            # =================================================
            # 7. GENERATE PDF REPORT
            # =================================================

            self.status_label.config(
                text="7/7 Generating PDF report..."
            )

            self.root.update_idletasks()

            report_path = os.path.join(
                output_dir,
                "Regression_Analysis_Report.pdf"
            )

            generate_report(
                output_path=report_path,
                dataset_name=dataset_name,
                target=target,
                validation=validation,
                cleaning_log=cleaning_log,
                eda_summary=eda_summary,
                target_plot=target_plot,
                correlation_plot=correlation_plot,
                scatter_plots=scatter_plots,
                pair_plot=pair_plot,
                comparison_df=comparison_df,
                overfitting_results=overfitting_results,
                residual_plots=residual_plots
            )

            # =================================================
            # ANALYSIS COMPLETED
            # =================================================

            self.status_label.config(
                text="✓ Analysis completed successfully!"
            )

            best_model = (
                comparison_df.iloc[0]["Model"]
            )

            best_score = (
                comparison_df.iloc[0]["Test R2"]
            )

            messagebox.showinfo(
                "Analysis Complete",
                f"Regression analysis completed successfully!\n\n"
                f"Highest Test R²:\n"
                f"{best_model} = {best_score:.4f}\n\n"
                f"PDF Report:\n"
                f"{report_path}"
            )

        except Exception as e:

            self.status_label.config(
                text="✗ Analysis failed."
            )

            messagebox.showerror(
                "Analysis Error",
                str(e)
            )

        finally:

            self.start_button.config(
                state="normal"
            )


# =====================================================
# RUN APPLICATION
# =====================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = RegressionAnalyzer(
        root
    )

    root.mainloop()