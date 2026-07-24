# ✈️ Flight Fare Prediction using Machine Learning

## 📌 Project Overview

Flight ticket prices fluctuate based on several factors such as airline, source, destination, departure time, duration, number of stops, and seasonality. This project develops a machine learning model to predict flight fares using historical flight data.

The project follows an end-to-end machine learning pipeline, including data preprocessing, feature engineering, model training, evaluation, and deployment-ready prediction.

---

# 🎯 Business Objective

Airfare prediction helps:

- Travelers estimate ticket prices before booking.
- Travel agencies provide fare recommendations.
- Airlines analyze pricing trends.
- Businesses optimize travel planning and budgeting.

The objective is to accurately predict the expected flight fare based on flight details.

---

# 📂 Dataset

The dataset contains historical flight booking information with features such as:

| Feature | Description |
|----------|-------------|
| Airline | Airline operating the flight |
| Date_of_Journey | Journey date |
| Source | Departure city |
| Destination | Arrival city |
| Route | Flight route |
| Dep_Time | Departure time |
| Arrival_Time | Arrival time |
| Duration | Total flight duration |
| Total_Stops | Number of stops |
| Additional_Info | Additional flight information |
| Price | Target variable (Flight Fare) |

---

# 🏗️ Project Structure

```
flight_fare_prediction/

│
├── model/
│   ├── flight_price_model_business.pkl
│   ├── flight_price_model_economy.pkl
│
├── notebook/
│   └── Flight_Fare_Business.ipynb
│   ├──  Flight_Fare_Economy.ipynb
│
├── app.py
├── requirements.txt
├── README.md

```

---

# 🔄 Project Workflow

## 1. Data Ingestion

- Load dataset
- Split data into business and economy class
- Store processed data

---

## 2. Data Preprocessing

The preprocessing pipeline includes:

- No Missing values in the dataset
- Remove and detect outliers in duration and price columns
- Remove column flight,as it doesnot contain any meaningful information
- Encoding categorical variables

Implemented using Scikit-learn Pipelines and ColumnTransformer.

---

## 3. Feature Engineering

Features engineered include:

- Airline
- Source_city
- Departure_time
- stops
- Arrival_time
- destination_city
- class
- duration
- days_left

---

## 4. Model Training
Trained two separate models for business class and economy class each
Multiple regression algorithms were evaluated, including:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

The best-performing model was selected based on evaluation metrics.
The best performing model was XGBoost Regressor
---

# 📊 Model Evaluation

Regression metrics used:

- R² Score for economy class - 0.9111
-  R² Score for business class - 0.9052

The final trained model provides accurate fare predictions for unseen flight data.

---

# 🚀 Prediction Pipeline

The prediction pipeline performs:

- Accept user flight details
- Apply the saved preprocessing pipeline
- Generate fare prediction using the trained model
- Return predicted ticket price

---

# 🌐 Deployment

The application is designed for deployment using stramlit and provides a simple web interface where users can enter flight details and receive fare predictions instantly.

---

# 🛠️ Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- CatBoost
- XGBoost
- Flask

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

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Start the Stramlit application:

```bash
stramlit run app.py
```

Open your browser and visit:

```
http://localhost:5000
```

Enter the required flight details and get the predicted airfare.

---
#Deployment
- Deployed on AWS-EC2
---
# 📈 Future Improvements

- Add real-time flight data integration

---

# 💻 Skills Demonstrated

- End-to-End Machine Learning Pipeline
- Feature Engineering
- Data Preprocessing
- Regression Modeling
- Model Evaluation
- Pipeline Serialization
- Scikit-learn Pipelines
- Git & GitHub

---

# 👨‍💻 Author

**Kunal Patil**

Machine Learning | Data Science | Automation Engineer

---

## ⭐ If you found this project useful, consider giving it a star on GitHub!
