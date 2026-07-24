
# 💳 Fraud Detection Using Machine Learning

## 📌 Project Overview

Financial fraud has become one of the biggest challenges in digital transactions. This project builds an end-to-end machine learning solution to detect fraudulent financial transactions in real time using historical transaction data.

The project covers the complete machine learning lifecycle, including data ingestion, preprocessing, feature engineering, model training, evaluation, and deployment through stramlit.

---

# 🎯 Business Objective

Financial institutions process millions of transactions every day, making manual fraud detection impossible. The objective of this project is to build a highly accurate fraud detection model that can identify suspicious transactions while minimizing false positives.

The solution can help:

- Detect fraudulent transactions in real time
- Reduce financial losses
- Improve customer trust
- Assist fraud investigation teams
- Automate fraud screening

---

# 📂 Dataset

The project uses a synthetic financial transaction dataset containing over **6.3 million transactions**.

### Dataset Features

| Feature | Description |
|----------|-------------|
| step | Time step (1 step = 1 hour) |
| type | Transaction type |
| amount | Transaction amount |
| nameOrig | Sender account |
| oldbalanceOrg | Sender balance before transaction |
| newbalanceOrig | Sender balance after transaction |
| nameDest | Receiver account |
| oldbalanceDest | Receiver balance before transaction |
| newbalanceDest | Receiver balance after transaction |
| isFraud | Target variable |
| isFlaggedFraud | Flagged transaction indicator |

---

# 📊 Exploratory Data Analysis

Performed extensive EDA to understand transaction patterns and fraud behavior.

Analysis included:

- Class imbalance analysis
- Fraud distribution
- Transaction type analysis
- Amount distribution
- Correlation analysis
- Missing value detection
- Outlier analysis

---

# 🏗️ Project Structure

```
fraud_detection/

│
├── models/
│   ├── fraud_pipeline.pkl
│
├── notebook/
│   └── Classification_Fraud_detection.ipynb
│
├── src/
│   ├── batch_predict.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── train.py
│
├── app.py
├── requirements.txt
├── README.md
```

---

# 🔄 Project Workflow

## 1. Data Ingestion

- Load transaction dataset
- Data validation
- Train-test split

---

## 2. Data Preprocessing

The preprocessing pipeline includes:

- Handling missing values
- Feature selection
- Encoding categorical variables
- Feature scaling
- Pipeline creation using Scikit-learn

---

## 3. Feature Engineering

Engineered features include:

- Balance difference
- Transaction amount features
- Missing balance indicators
- Encoded transaction types
- Customer transaction behavior features

---

## 4. Model Training

Several machine learning models were evaluated:

- Random Forest Classifier
- XGBoost Classifier

Hyperparameter tuning was performed to improve model performance.

The best-performing model was selected based on fraud detection metrics.
The best performing model was XGBoost Classifier
---

# 📈 Model Performance

The final XGBoost model achieved approximately:

| Metric | Score |
|---------|-------|
| Precision | **97%** |
| Recall | **80%** |
| F1 Score | **88%** |
| ROC-AUC | **99%** |

These results indicate a strong balance between identifying fraudulent transactions and minimizing false alarms.

---

# 🚀 Real-Time Prediction API

Used stramlit to prepare an interactive interface
Give inputs and it will predict whether the transaction is fraudulent or not.


Deployment architecture:


# 🛠️ Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Joblib

---

# 📦 Installation

Clone the repository:

```bash
git clone <repository-url>
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Start the API server:

```bash
streamlit run app.py
```

Use the `Predict Fraud` button to test fraud.

---

# 📈 Future Improvements

- Deploy on AWS using Docker

---

# 💻 Skills Demonstrated

- End-to-End Machine Learning Pipeline
- Feature Engineering
- Data Preprocessing
- Imbalanced Data Analysis
- Classification Modeling
- Hyperparameter Tuning
- Model Evaluation
- Python OOP
- Scikit-learn Pipelines
- XGBoost
- Git & GitHub

---

# 👨‍💻 Author

**Kunal Patil**

Machine Learning | Data Science | Automation Engineer

---

## ⭐ If you found this project useful, consider giving it a star on GitHub!
