# Regression Model Analyzer

## Overview
Explore and clean tabular data, select a target variable, inspect correlations, compare regression models, and review actual-vs-predicted and residual diagnostics.

## Streamlit UI setup

**Prerequisites:** Python installed.

Open a terminal in this project directory and create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

**PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Command Prompt (CMD)**
```bat
.venv\Scripts\activate.bat
```

Install the Streamlit app dependencies:

```bash
pip install -r requirements_streamlit.txt
```

Start the app:

```bash
streamlit run streamlit_app.py
```

Streamlit prints the local address (typically `http://localhost:8501`) and opens the UI in your browser.

## Original terminal version
The original source files are retained. Run the terminal entry point from this folder:

```bash
python main.py
```

If required, install the original project's dependencies from its original requirements file. `requirements_streamlit.txt` covers the UI version.

## Features
CSV upload and preview; data-quality checks; target selection; correlation exploration; regression-model comparison; performance metrics; actual-vs-predicted and residual plots.

## Dataset requirements
Upload a CSV. Select a numeric target variable; other columns are used as predictors.

## Notes
- Run commands from the project directory to ensure relative paths work.
- Keep the original scripts and data files in place.
- This Streamlit app runs locally; it is not automatically deployed online.
