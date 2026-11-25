# Credit Risk ML Project

An exploratory machine learning project for predicting the probability that a loan applicant will default. The project combines a public-style credit-risk dataset, exploratory analysis, feature preprocessing, a Random Forest model, and an optional MLflow/MinIO/MySQL tracking stack.

> **Project status:** the repository is in an early development stage. The EDA notebook is the primary implemented analysis. Several pipeline and packaging files are currently incomplete or contain known implementation mismatches; see [Known development gaps](#known-development-gaps).

## Project overview

The target variable is `loan_status`:

- `0` indicates a non-default or paid loan.
- `1` indicates a default or risky loan.

The current dataset contains 32,581 records and 12 columns, including demographic, employment, loan, credit-history, and default-history features.

| Feature | Description |
|---|---|
| `person_age` | Applicant age |
| `person_income` | Annual income |
| `person_home_ownership` | Home ownership status |
| `person_emp_length` | Employment length in years |
| `loan_intent` | Purpose of the loan |
| `loan_grade` | Assigned loan grade |
| `loan_amnt` | Loan amount |
| `loan_int_rate` | Interest rate |
| `loan_percent_income` | Loan amount as a percentage of income |
| `cb_person_default_on_file` | Historical default record |
| `cb_person_cred_hist_length` | Credit-history length |
| `loan_status` | Target variable |

## Repository structure

```text
.
|-- data/
|   |-- credit_risk_dataset.csv
|   `-- EDA CREDIT-RISK-DETECTION.ipynb
|-- logs/                       # Timestamped application log directories
|-- src/
|   |-- exception.py            # Custom exception helper
|   |-- logger.py               # Logging configuration
|   `-- pipeline/
|       |-- data_transformation.py
|       `-- data_training.py
|-- docker-compose.yml          # MLflow, MySQL, and MinIO services
|-- Dockerfile
|-- requirements.txt
`-- setup.py
```

## Data preprocessing direction

The intended preprocessing workflow includes:

- Reading and inspecting the credit-risk dataset.
- Handling missing or invalid values.
- Encoding `person_home_ownership` with one-hot encoding.
- Ordinal encoding of `loan_grade`.
- Grouping `person_emp_length` into employment-length ranges.
- Splitting the data into training and test sets.
- Training a `RandomForestClassifier`.
- Evaluating the model with accuracy, precision, recall, F1 score, and ROC AUC.
- Tracking experiments and artifacts with MLflow.

## Getting started

### Prerequisites

- Python 3.10 or later
- pip
- Jupyter Notebook or JupyterLab
- Docker and Docker Compose for the optional MLflow infrastructure

### Local setup

From the project root:

```bash
python -m venv .venv
```

Activate the environment:

```powershell
.venv\Scripts\Activate.ps1
```

Install the current dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install jupyter
```

Start the EDA notebook:

```bash
python -m jupyter notebook "data/EDA CREDIT-RISK-DETECTION.ipynb"
```

## MLflow infrastructure

The Docker Compose configuration defines:

- MLflow server on port `5000`
- MySQL on port `3306`
- MinIO API on port `9000`
- MinIO web console on port `9001`

To start the stack:

```bash
docker compose up --build
```

The MLflow UI is then available at:

```text
http://localhost:5000
```

## Running code

The package and pipeline files are still being developed. At present, the notebook is the most complete runnable workflow.

Logging is configured by `src/logger.py`; log files are written beneath `logs/` in timestamped directories.

## Known development gaps

The following issues should be resolved before treating the repository as a complete training pipeline:

- `data_transformation.py` is incomplete and references `data/credit_risk.csv`, but the repository dataset is named `data/credit_risk_dataset.csv`.
- `data_training.py` contains an import typo: `RandomForestXlassifier` instead of `RandomForestClassifier`.
- The split return statements in `data_transformation.py` need to be placed inside the intended function scope.
- `data_transformation.py` attempts `df["person_home_ownership"].value_counts()` after converting the column into dummy variables.
- `setup.py` and `requirements.txt` are incomplete; MLflow, PyMySQL, and object-storage dependencies required by the Docker stack are not declared.
- The Dockerfile installs requirements before copying the project files, so it needs to copy `requirements.txt` before running `pip install`.
- The Compose MLflow image requires the MLflow server package and database/object-storage support to be installed.
- The project does not currently include a complete CLI entry point or test suite.

## Intended model evaluation

For binary credit-risk classification, recommended evaluation metrics include:

- **Accuracy** for overall correctness
- **Precision** for false-positive control
- **Recall** for identifying as many risky loans as possible
- **F1 score** for balancing precision and recall
- **ROC AUC** for ranking performance across thresholds
- **Confusion matrix** for understanding the types of prediction errors

## License

No license is currently included in this repository. Add a `LICENSE` file if third-party or commercial use is intended.
