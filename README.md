# CreditCardDefault_End_to_End_ML_Capstone_Project

## Problem
Predict whether a credit card customer will default on payment in the next month based on demographic, financial, and repayment history data. This is a supervised binary classification problem demonstrating an end-to-end machine learning workflow including EDA, feature engineering, model training, evaluation, and deployment.

---

## Badges

[![Dataset](https://img.shields.io/badge/dataset-UCI%20Credit%20Card%20Default-blue)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-green)]()

---

## Table of contents

* Project overview  
* Dataset  
* Exploratory Data Analysis (EDA)  
* Modeling  
* Results & selected model  
* How to run (local)  
* API — serving predictions  
* Docker  
* Directory structure  

---

## Project overview

**Problem.**  
Banks and financial institutions face significant losses due to credit card defaults. The objective of this project is to predict whether a customer will default on their credit card payment in the next billing cycle using historical customer data.

**Goal.**  
Build a complete, reproducible ML pipeline covering data analysis, preprocessing, model selection, evaluation, model persistence, and deployment as a REST API using FastAPI and Docker.

**Why this dataset?**  
The UCI Credit Card Default dataset is a real-world, imbalanced classification dataset widely used in industry and academia. It captures both behavioral and demographic signals and is well-suited for demonstrating practical ML decision-making.

---

## Dataset

**Source:** UCI Machine Learning Repository — Credit Card Default Dataset.

Each row represents a customer, with features describing credit limit, demographics, past payment behavior, bill amounts, and payment amounts.

**Target variable:**

* `default.payment.next.month`
  * `1` → Default
  * `0` → No Default

**Key feature groups:**

* **Demographics:** SEX, EDUCATION, MARRIAGE, AGE  
* **Credit information:** LIMIT_BAL  
* **Repayment history:** PAY_0 to PAY_6  
* **Billing amounts:** BILL_AMT1 to BILL_AMT6  
* **Payment amounts:** PAY_AMT1 to PAY_AMT6  

---

## Exploratory Data Analysis (EDA)

EDA is performed in `notebook.ipynb` and includes:

* Dataset overview (`shape`, `info`, `describe`)
* Missing value and invalid category checks
* Target class imbalance analysis
* Distribution plots for numerical features
* Categorical feature frequency analysis
* Correlation matrix and heatmap
* Relationship between repayment behavior and default probability

Key observations from EDA guide feature selection and model choice.

---

## Modeling

Multiple classification models are trained and compared:

1. Logistic Regression — interpretable baseline
2. Decision Tree Classifier — non-linear baseline
3. Random Forest Classifier — ensemble model (primary focus)

**Validation strategy:**

* Train / validation / test split  
  * Train: 64%  
  * Validation: 16%  
  * Test: 20%
* Stratified splitting to preserve class balance
* Evaluation metrics:
  * Accuracy
  * Precision
  * Recall
  * F1-score
  * ROC-AUC

**Hyperparameter tuning:**

* GridSearchCV applied to Random Forest
* Tuned parameters include:
  * `n_estimators`
  * `max_depth`
  * `min_samples_split`

---

## Results & selected model

* Models evaluated: Logistic Regression, Decision Tree, Random Forest
* Primary metric: ROC-AUC on validation set

**Final model selection:**

* Random Forest achieved the best balance between recall and ROC-AUC
* Final model retrained on combined train + validation data
* Tested on a held-out test set for unbiased performance estimation

**Artifacts produced:**

* `model.joblib` — serialized trained model
* Evaluation plots:
  * Confusion matrix
  * ROC curve
  * Feature importance bar chart

---

## How to run (local)

1. Setup

```
python -m venv .venv
source .venv/bin/activate      
pip install -r requirements.txt
```

2. Download dataset

```
curl -o data/credit_card_default.xls
"https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"

```

Run the notebook to convert XLS → CSV.

3. Run the notebook

Open notebook.ipynb and run all cells to reproduce EDA and experiments.

4. Train & save final model (script)

```
python train.py

```

5. Run the API (local)

```
uvicorn predict:app --host 0.0.0.0 --port 8000

```

Test with curl:

```
curl -X POST "http://127.0.0.1:8000/predict"
-H "Content-Type: application/json"
-d '{"limit_bal":200000,"sex":2,"education":2,"marriage":1,"age":35,
"pay_0":-1,"pay_2":0,"pay_3":0,"pay_4":0,"pay_5":0,"pay_6":0}'

```

---

## API — Serving predictions (predict.py)

* Implemented with FastAPI.
* Endpoint GET / returns health status.
* Endpoint POST /predict returns probability of default.

Response example:

```
{"default_probability": 0.27}

```

---

## Docker

Build and run the API in Docker.

```
docker build -t credit-default-ml:v1 .
docker run -p 8000:8000 credit-default-ml:v1
```

This image installs dependencies from requirements.txt, copies code, exposes port 8000 and runs uvicorn predict:app.

---

## Directory structure

```
├── data/
│ └── credit_card_default.csv
├── notebook.ipynb
├── train.py
├── predict.py
├── model.joblib
├── requirements.txt
├── Dockerfile
└── README.md
```