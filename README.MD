Predict the Behavior of Customers

Short description
- Notebook-driven project to analyze online retail transactions and build models that predict whether a customer will make a purchase within the next quarter (90 days). The primary implementation and pipeline are in `main.ipynb`.

Key features and functionalities
- Data ingestion and cleaning (reads `online_retail_II.csv`).
- Feature engineering: computes Recency, Frequency, Revenue (RFM).
- RFM clustering (K-Means) and ordering of clusters.
- Creation of a binary target `WillBuyInNext90Days` (1 if next purchase ≤ 90 days).
- Model training and evaluation (Logistic Regression, XGBoost, Random Forest, SVM, KNN, Naive Bayes, Decision Tree).
- Hyperparameter search with GridSearchCV for XGBoost and Logistic Regression.
- Visualizations using Plotly and seaborn (histograms, elbow plots, confusion matrix heatmaps).

Repository contents (relevant)
- `main.ipynb` — main Jupyter notebook with the full pipeline: data engineering, feature engineering, clustering, modeling and evaluation.
- `online_retail_II.csv` — dataset expected by `main.ipynb` (place this file next to the notebook).

Requirements and dependencies
- Python 3.8+ recommended
- Major Python packages used in the notebook:
  - pandas, numpy, matplotlib, plotly, seaborn
  - scikit-learn
  - xgboost
  - multiscorer
- Example pip install (recommended in a virtual environment):
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install pandas numpy matplotlib plotly seaborn scikit-learn xgboost multiscorer jupyter
```

Installation and setup (step-by-step)
1. Clone or copy the project folder to your machine.
2. Make sure `main.ipynb` and `online_retail_II.csv` are in the same folder.
3. Create and activate a virtual environment (Windows example):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
4. Install dependencies:
```powershell
pip install pandas numpy matplotlib plotly seaborn scikit-learn xgboost multiscorer jupyter
```
5. Start Jupyter or open the notebook in VS Code:
```powershell
jupyter notebook main.ipynb
# or open VS Code and run the notebook cells
```

Usage examples
- Typical usage is to open `main.ipynb` and run cells in order. The notebook performs:
  - data loading: `df = pd.read_csv('online_retail_II.csv')`
  - basic cleaning & datetime conversion
  - split into behavior / next-quarter datasets
  - compute RFM features and clusters
  - create target: `ctm_dt['WillBuyInNext90Days']`
  - train models via the packaged `train_and_evaluate_models` function in the notebook

Minimal script fragment (adapted from the notebook) to preview data:
```python
import pandas as pd
df = pd.read_csv("online_retail_II.csv")
df.rename(columns={'Invoice':'InvoiceNo', 'Customer ID':'CustomerID', 'Price':'UnitPrice'}, inplace=True)
df = df.dropna()
df['InvoiceDate'] = pd.to_datetime(df.InvoiceDate)
print(df.info())
```

Notes on running the ML pipeline
- The notebook uses fixed random_state seeds for reproducibility in clustering and model splits (e.g., `random_state=42`).
- Grid searches (XGBoost) may be compute-intensive; adjust `cv` / parameter ranges for local machines.
- Visual output uses Plotly and seaborn; run interactively in Jupyter / VS Code.

Contribution guidelines
- This repository is notebook-first. Contributions are welcome as:
  - Issues describing bugs, missing data or suggestions.
  - Pull requests that add modular Python scripts or tests (keep notebook changes minimal and include runnable code cells).
- When contributing:
  - Add a clear description of changes.
  - If adding dependencies, update the README and include a short rationale.
  - Prefer reproducible examples and small, focused commits.

Additional notes and contact
- For questions, issues, or suggestions, open an issue in the repository or contact the project owner (repository maintainer).
- For reproducible runs, consider exporting a `requirements.txt` after installing dependencies:
```powershell
pip freeze > requirements.txt
```
- If desired, the notebook can be refactored into a Python package or script for non-interactive batch runs.