# AI Impact on Jobs & Salaries (2020–2026)

An exploratory Data Science and Machine Learning project investigating occupational exposure to Generative AI and its relationship with salaries.

## Project Overview

Generative AI is transforming the labor market and changing how tasks are performed across different occupations.

This project combines occupational AI exposure scores with salary data to investigate two main research questions:

1. Which occupations are more exposed to Generative AI?
2. Does occupational AI exposure provide additional predictive information for salary estimation?

The project applies Exploratory Data Analysis (EDA), Feature Engineering, Natural Language Processing (NLP), and Machine Learning regression techniques to explore these questions.

**Important:** AI exposure does not necessarily imply job displacement. This project investigates occupational exposure and statistical associations, not causal effects of AI on employment or salaries.

---

## Project Structure

```text
AI_Impact_on_Jobs&Salaries/
│
├── data/
│   ├── ai_jobs_salaries_clean.csv
│   ├── ai_jobs_salaries_with_exposure.csv
│   ├── data_dictionary.md
│   └── Final_Scores_ISCO08_Gmyrek_et_al_2025.xlsx
│
├── models/
│   ├── ai_exposure_ridge.joblib
│   ├── salary_model_with_exposure.joblib
│   └── salary_model_without_exposure.joblib
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_salary_modeling.ipynb
│
├── reports/
│   └── salary_model_comparison.csv
│
└── README.md
```

---

## Datasets

This project uses two main datasets.

### 1. AI Jobs and Salaries Dataset

Source: [Kaggle – AI Impact on Jobs and Salaries (2020–2026)](https://www.kaggle.com/datasets/debayank2024/ai-impact-on-jobs-and-salaries-2020-2026)

The dataset contains information about salaries and employment characteristics, including:

- Work year
- Job title
- Experience level
- Employment type
- Salary in USD
- Work mode
- Company size
- Role family
- Approximate ISCO occupation group

Original dataset shape: **71,913 rows and 17 columns**.

### 2. AI Occupational Exposure Dataset

Source: [International Labour Organization (ILO)](https://www.ilo.org/publications/generative-ai-and-jobs-refined-global-index-occupational-exposure)

Dataset: [Gmyrek et al. (2025) – ISCO-08 AI Exposure Scores](https://github.com/pgmyrek/2025_GenAI_scores_ISCO08)

This dataset provides task-level and occupation-level information about exposure to Generative AI.

Important variables include:

- ISCO-08 occupation codes
- Occupation titles
- Task descriptions
- 2023 exposure scores
- 2025 exposure scores
- Mean occupation-level exposure scores
- Exposure categories

The dataset contains **427 unique occupations** after occupation-level aggregation.

### Data Integration

The salary dataset was combined with the occupational AI exposure dataset using available occupational classification information.

The resulting file is:

`data/ai_jobs_salaries_with_exposure.csv`

The merged dataset contains **28,521 records** with valid AI exposure scores.

**Important limitation:** Only four unique ISCO occupation groups and four distinct AI exposure scores are represented in the merged salary dataset. Therefore, the merged dataset does not provide comprehensive occupational coverage.

---

## Project Workflow

The project is organized into four Jupyter notebooks.

| Notebook | Description |
|---|---|
| `01_eda.ipynb` | Exploratory Data Analysis |
| `02_feature_engineering.ipynb` | Data preparation and feature engineering |
| `03_modeling.ipynb` | AI exposure prediction |
| `04_salary_modeling.ipynb` | Salary prediction and model comparison |

---

## Model 1: Predicting Occupational AI Exposure

### Objective

Predict the level of occupational exposure to Generative AI using job titles and task descriptions.

**Research question:**

Which occupations are more exposed to Generative AI?

### Dataset Preparation

- 427 unique occupations
- Occupation-level aggregated task descriptions
- Target variable: `mean_score_2025`
- 58 unique target score values

### Feature Engineering

Occupation titles and task descriptions were combined into a text feature.

TF-IDF Vectorization was used to transform textual information into numerical features.

### Algorithm: Ridge Regression

Ridge Regression was selected to predict continuous AI exposure scores.

L2 regularization helps control model complexity when working with high-dimensional TF-IDF features.

Cross-validation and GridSearchCV were used for hyperparameter tuning.

### Model Evaluation

| Metric | Test Result |
|---|---:|
| MAE | 0.0372 |
| RMSE | 0.0529 |
| R² | 0.8530 |

The model achieved an R² score of 0.8530 on the held-out test set, indicating promising predictive performance on the available occupation-level dataset.

---

## Model 2: AI Exposure and Salary Prediction

### Objective

Investigate whether occupational AI exposure provides additional predictive information for salary estimation.

**Research question:**

Does occupational AI exposure improve salary prediction when considered alongside employment-related features?

### Target Variable

`salary_in_usd`

A logarithmic transformation was applied to salary because the original salary distribution was right-skewed.

### Model Comparison

Two Ridge Regression models were developed.

**Model A — Without AI Exposure**

Uses employment-related features without the AI exposure score.

**Model B — With AI Exposure**

Uses the same features with the addition of `mean_score_2025`.

The `role_family` feature was excluded from the final comparison because it perfectly determined the exposure score within the available merged dataset.

### Test Set Results

| Metric | Model A | Model B |
|---|---:|---:|
| MAE (USD) | 50,181.74 | 48,704.59 |
| RMSE (USD) | 69,582.94 | 67,953.05 |
| R² | 0.1049 | 0.1464 |

### Cross-Validation Results

Five-fold cross-validation was used to compare both models.

| Metric | Model A | Model B |
|---|---:|---:|
| CV MAE (log scale) | 0.3751 | 0.3634 |

### Interpretation

Adding AI exposure resulted in:

- Approximately 2.94% lower test MAE.
- Approximately 2.35% lower test RMSE.
- An increase in R² from 0.1049 to 0.1464.
- Approximately 3.1% lower cross-validation MAE.

These findings suggest that AI exposure provides modest additional predictive information for salary estimation within this dataset.

However, overall salary prediction performance remains limited.

---

## Key Findings

1. Occupational exposure to Generative AI varies across occupations.
2. TF-IDF combined with Ridge Regression achieved promising results in predicting occupation-level AI exposure.
3. Adding AI exposure modestly improved salary prediction performance.
4. The available merged salary dataset contains only four unique ISCO groups and four distinct AI exposure scores.
5. The results identify predictive associations and should not be interpreted as evidence that AI directly increases or decreases salaries.

---

## Limitations

Several limitations should be considered when interpreting the findings:

- **Limited occupational coverage:** Only four unique ISCO occupation groups are represented in the merged salary dataset.
- **Limited exposure variation:** Only four distinct AI exposure scores are available for salary modeling.
- **Approximate occupation mapping:** The occupation mapping in the salary dataset is approximate.
- **Salary variability:** Salary levels may depend on factors such as country, experience, industry, and company characteristics.
- **Non-causal analysis:** The models investigate predictive relationships, not causal effects of AI on salaries.
- **Generalization:** Random train-test splitting may not reliably measure performance on previously unseen occupational groups.

The findings should therefore be considered exploratory rather than definitive evidence about the labor market.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- TF-IDF Vectorization
- Ridge Regression
- Cross-validation
- GridSearchCV
- Jupyter Notebook
- Joblib

---

## Model Files

Trained models are stored in the `models/` directory.

| File | Description |
|---|---|
| `ai_exposure_ridge.joblib` | AI occupational exposure prediction pipeline |
| `salary_model_without_exposure.joblib` | Salary prediction model without AI exposure |
| `salary_model_with_exposure.joblib` | Salary prediction model with AI exposure |

The salary model comparison results are available in:

`reports/salary_model_comparison.csv`

---

## Installation and Usage

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AI_Impact_on_Jobs_Salaries
```

### 2. Install dependencies

Install the required Python libraries used in the notebooks:

```bash
pip install pandas numpy matplotlib scikit-learn openpyxl joblib jupyter
```

### 3. Open the notebooks

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the `notebooks/` directory and run the notebooks in the following order:

1. `01_eda.ipynb`
2. `02_feature_engineering.ipynb`
3. `03_modeling.ipynb`
4. `04_salary_modeling.ipynb`

---

## Future Improvements

Potential future improvements include:

- Integrating a broader range of occupations using reliable ISCO-08 mappings.
- Increasing the diversity of AI exposure scores in the salary dataset.
- Evaluating models using occupation-based group splits.
- Investigating salary trends across multiple years.
- Exploring additional regression algorithms.
- Analyzing AI exposure across different industries and experience levels.
- Using more representative and longitudinal salary data.

---

## Data Sources and Attribution

- Salary dataset: [Kaggle – AI Impact on Jobs and Salaries](https://www.kaggle.com/datasets/debayank2024/ai-impact-on-jobs-and-salaries-2020-2026)
- AI exposure research: [International Labour Organization (ILO)](https://www.ilo.org/publications/generative-ai-and-jobs-refined-global-index-occupational-exposure)
- AI exposure dataset: [Gmyrek et al. (2025) – GitHub Repository](https://github.com/pgmyrek/2025_GenAI_scores_ISCO08)

Please refer to the original sources for their respective licensing and attribution requirements.

---

## Author

**Newsha**

AI Programmer | Machine Learning | Data Science

---

*This project explores occupational exposure to Generative AI and its relationship with salaries using data analysis and machine learning. All findings should be interpreted within the limitations of the available datasets.*