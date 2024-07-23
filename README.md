# Car-Price-Prediction


## Overview
This repository contains code for predicting car prices based on various features using machine learning models.

## Dataset
The dataset used for this project can be found [here](https://www.kaggle.com/datasets/CooperUnion/cardataset).

## Project Structure
- **data/**: Directory for datasets.
- **notebooks/**: Jupyter notebooks for EDA, model building, and evaluation.
- **src/**: Source code for data preprocessing, model training, and evaluation.
- **models/**: Saved models.
- **scripts/**: Scripts for running the model and API.

## Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/car-price-prediction.git


    Data Collection
Ensure you have a dataset, car_data.csv, which includes features like car model, year, mileage, price, etc.

   Data Preprocessing
Create a Python script (data_preprocessing.py) to handle data cleaning and preprocessing.

#code;
# src/data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    # Handle missing values
    df = df.dropna()
    
    # Convert categorical data to numerical data
    df = pd.get_dummies(df, drop_first=True)
    
    # Split data into features and target
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_data('../data/car_data.csv')
    X_train, X_test, y_train, y_test = preprocess_data(df)
    # Save the preprocessed data
    pd.DataFrame(X_train).to_csv('../data/X_train.csv', index=False)
    pd.DataFrame(X_test).to_csv('../data/X_test.csv', index=False)
    pd.DataFrame(y_train).to_csv('../data/y_train.csv', index=False)
    pd.DataFrame(y_test).to_csv('../data/y_test.csv', index=False)

Model Training:

# src/model_training.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def load_preprocessed_data():
    X_train = pd.read_csv('../data/X_train.csv')
    X_test = pd.read_csv('../data/X_test.csv')
    y_train = pd.read_csv('../data/y_train.csv')
    y_test = pd.read_csv('../data/y_test.csv')
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    model = train_model(X_train, y_train)
    mse = evaluate_model(model, X_test, y_test)
    print(f'Mean Squared Error: {mse}')

Utility Functions:

# src/utils.py
import pandas as pd

def load_csv(file_path):
    return pd.read_csv(file_path)

