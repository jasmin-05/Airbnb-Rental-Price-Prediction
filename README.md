# Airbnb-Rental-Price-Prediction
Internship project: Predicting Airbnb rental prices using machine learning regression techniques (Linear Regression, Random Forest, XGBoost). Includes data preprocessing, feature engineering, model evaluation, and results visualization.
🏠 Airbnb Rental Price Prediction
Predicting Airbnb rental prices using machine learning

This project aims to build a machine learning model that predicts Airbnb listing prices based on features such as location, number of rooms, host rating, property type, and more.
It includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and saving final results.

📁 Project Structure
Airbnb-Rental-Price-Prediction/
│
├── src/
│   ├── model_training.py       # Main ML training script
│   └── notebooks/              # Jupyter notebooks folder
│
├── notebooks/
│   ├── exploratory.ipynb       # EDA notebook
│
├── data/
│   ├── raw/                    # Raw dataset
│   └── processed/              # Cleaned dataset
│
├── results/
│   ├── models/                 # Saved ML models
│   └── figures/                # Graphs and charts
│
├── requirements.txt
├── README.md
└── .gitignore

🚀 Features

Cleaned and preprocessed Airbnb listing dataset

Exploratory Data Analysis (EDA)

Feature engineering

ML model building using:

Linear Regression

Random Forest Regressor

Gradient Boosting

Model evaluation:

MAE

RMSE

R² score

Saving trained models for reuse

🛠️ Tech Stack

Python

Pandas, NumPy

Scikit-learn

Matplotlib / Seaborn

Jupyter Notebook

▶️ How to Run the Project
1. Clone the repository
git clone https://github.com/<your-username>/Airbnb-Rental-Price-Prediction.git

2. Install dependencies
pip install -r requirements.txt

3. Add your dataset

Place your dataset in:

data/raw/


Example file:
data/raw/airbnb_listings.csv

4. Run the training script
python src/model_training.py


This will:

preprocess the dataset

train ML models

evaluate them

save the final model to:

results/models/

📊 Results

After training, you will get:

Model performance metrics (MAE, RMSE, R²)

Saved trained model

Feature importance plots

Data visualizations

📘 Notebooks

All exploratory analysis is included in:

notebooks/exploratory.ipynb

📝 Future Improvements

Hyperparameter tuning

Adding deep learning models

Deploying model as a web API

Integrating with a Streamlit dashboard
