<div align="center">

# > INSURANCE // RENEWAL <

**_`MACHINE LEARNING PROJECT: INSURANCE POLICY RENEWAL PREDICTION SYSTEM.`_**

<img src="/static/header.png" alt="logo" width="60%">
<img src="/static/mockups.png" alt="mockups">

<img src="https://img.shields.io/github/repo-size/sophie-muriel/ProyectoFinal-Muriel-Vitonco?style=for-the-badge&color=black&labelColor=grey" alt="Repo Size">
<img src="https://img.shields.io/github/last-commit/sophie-muriel/ProyectoFinal-Muriel-Vitonco?style=for-the-badge&color=ffe058&labelColor=black" alt="Last Commit">
<img src="https://img.shields.io/github/contributors/sophie-muriel/ProyectoFinal-Muriel-Vitonco?style=for-the-badge&color=black&labelColor=grey" alt="Contributors">

</div>

## ⚡ WEB APP // LIVE DEMO

This project runs as a Flask web application for real-time inference.
The production instance is statically deployed on **Railway**.

**🔗 ACCESS HERE** > **[insurance-renewal.up.railway.app](https://insurance-renewal.up.railway.app/)**

## ⚠️ NOTE: EXTERNAL FILES

To keep the repository lightweight and optimized, heavy binary files (`.pkl`) are **NOT** hosted here.

- **Source:** They are automatically generated in `[SECTION 10]` of the Jupyter Notebook when executed.
- **Runtime:** During deployment, the system automatically downloads the models from **Hugging Face Hub**.

| FILE                          | TYPE                  | LINK                                                                                                              |
| :---------------------------- | :-------------------- | :---------------------------------------------------------------------------------------------------------------- |
| `insurance_renewal_model.pkl` | Model (Random Forest) | [Hugging Face Repo](https://huggingface.co/sophie-muriel/insurance-renewal/blob/main/insurance_renewal_model.pkl) |
| `scaler.pkl`                  | Scaler (MinMax)       | [Hugging Face Repo](https://huggingface.co/sophie-muriel/insurance-renewal/blob/main/scaler.pkl)                  |

## 🧭 TABLE OF CONTENTS // NAVIGATION

1.  [PROJECT DETAILS](#-project-details)
2.  [FILE STRUCTURE](#-file-structure)
3.  [SETUP](#-setup)
4.  [EXECUTION INSTRUCTIONS](#-execution-instructions)
5.  [GENERAL CONCLUSIONS](#-general-conclusions)
6.  [AUTHORS](#-authors)

## 📘 PROJECT DETAILS

This repository analyzes behavior patterns in insurance policy payments to predict renewal probability using the provided dataset (`insurance_company.csv`). The workflow covers everything from raw data ingestion to inference deployment.

**/// NOTEBOOK.IPYNB COMPONENTS:**

- `BUSINESS CASE`: Introduction, problem identification (customer retention & financial impact), data, objectives, and variables (dependent/independent).
- `DESCRIPTION`: Library/data loading and general dataset & variable info.
- `EDA`: Exploratory data analysis (pattern & outlier detection) + Data Profiling.
- `PREPROCESSING`: Cleaning, encoding, imputation, normalization, and variable transformation.
- `MODELING`: Training and evaluation of models (Random Forest, KNN, Logistic Regression).
- `RECOMMENDATIONS`: Final conclusions and suggested actions to improve retention and optimize sales efforts.
- `DEPLOYMENT`: Flask API + styled frontend hosted on **Railway**.

## 📂 FILE STRUCTURE

```text
INSURANCE-RENEWAL/
│
├── data/                                    # [DATASET INPUT/OUTPUT]
│   ├── crosstabs/
│   ├── grouped_describe_by_renewal_cat.csv
│   ├── grouped_describe_by_renewal_num.csv
│   ├── insurance_company.csv
│   └── insurance_company_final.csv
│
├── images/                                  # [VISUALIZATION OUTPUTS]
│   ├── univariable/
│   ├── bivariable/
│   ├── models/
│   ├── corr_matrix_filtered.png
│   ├── corr_matrix.png
│   ├── renewal_dist.png
│   └── renewal_smote_dist.png
│
├── static/                                  # [FRONTEND STYLES + ASSETS]
│   ├── css/styles.css
│   ├── favicon.svg
│   ├── header.png
│   └── mockups.png
│
├── templates/                               # [WEB INTERFACE]
│   └── index.html
│
├── app.py                                   # [FLASK BACKEND]
├── notebook.ipynb                           # [JUPYTER NOTEBOOK]
├── eda_profiling_report.html                # [YDATA-PROFILING REPORT]
├── presentation.pdf                         # [PRESENTATION]
├── .python-version                          # [PYTHON VERSION]
├── requirements.txt                         # [DEPENDENCIES]
└── README.md                                # < YOU ARE HERE >
```

### 📝 GENERAL DESCRIPTION

- `data/`: Original/transformed datasets and statistical tables generated during EDA.
- `images/`: Visualizations from the analysis (univariate, bivariate, correlations, model plots).
- `static/`: CSS styles, mockup images (assets), and site favicon.
- `templates/`: HTML template for the web app (`index.html`).
- `app.py`: Main Flask server file — handles routes, model loading, and predictions.
- `notebook.ipynb`: EDA, data transformation, training, model evaluation, recommendations.
- `eda_profiling_report.html`: Auto-generated report using YData-Profiling.
- `requirements.txt`: All dependencies needed to reproduce the project.

## 🛠️ SETUP

**SYSTEM REQUIREMENTS:**

- Python 3.9+
- pip
- Virtual environment (recommended)
- Web browser
- Dependencies:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - ydata-profiling
  - scipy
  - imbalanced-learn
  - scikit-learn
  - flask
  - ipykernel
  - ipywidgets
  - huggingface_hub
  - gunicorn

**DEPENDENCIES INSTALLATION:**

```bash
pip install -r requirements.txt
```

## 🚀 EXECUTION INSTRUCTIONS

### 1. CLONE REPOSITORY

```bash
git clone https://github.com/sophie-muriel/insurance-renewal.git
cd insurance-renewal
```

### 2. CREATE VIRTUAL ENVIRONMENT (RECOMMENDED)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. INSTALL DEPENDENCIES

```bash
pip install -r requirements.txt
```

### 4. RUN NOTEBOOK

To re-run the full analysis and retrain models locally:

```bash
jupyter notebook
# Run all cells in 'notebook.ipynb'
```

### 5. RUN FLASK APP LOCALLY (INFERENCE)

```bash
python app.py
```

_The server will start at `http://localhost:8080`._

### 6. ... OR GO TO THE LIVE DEMO

Right here > [INSURANCE // RENEWAL](#-web-app--live-demo)

## 📊 GENERAL CONCLUSIONS

> \*Main goal: **What factors influence the likelihood of renewing insurance premiums, and how can the dataset be prepared, modeled, and evaluated to predict this probability and optimize incentives?\***

The final model (**Random Forest**) was chosen after EDA, data preprocessing, and comparative testing of multiple models — mainly because of its ability to handle severe class imbalance (~6.3% churn rate). This model also shows excellent differentiation between renewing and non-renewing customers.

**> MODEL STATUS:**

- **Target Variable:** Renewal (binary); `renewal`.
- **Priority:** Maximize `Recall` on the minority class (Non-Renewal).
- **Metrics**: High precision, outstanding F1-score, strong ROC-AUC with good class separation, and more.

> _Exact values can be found in `[SECTION 8.2]` of the Jupyter Notebook._

**> HIGHEST-IMPACT FEATURES:**

1. `perc_premium_paid_by_cash_credit`
2. `income`
3. `application_underwriting_score`
4. `age_in_years`
5. `total_late_payments`
6. `has_late_payments` (history)

For the full analysis, check `presentation.pdf` in the repo — these slides contain the complete project summary, detailed methodology, visual findings, and final strategic conclusions. It is highly recommended to review it for a comprehensive understanding of the business case and all key insights.

## 👥 AUTHORS

Made by:

- **Sophie Muriel** > [GITHUB PROFILE](https://github.com/sophie-muriel)
- **Karol Vitonco** > [GITHUB PROFILE](https://github.com/KrlVanessa)
