# Daily Weather Forecasting using Recurrent Neural Networks (RNN)

This project demonstrates the use of a Long Short-Term Memory (LSTM) based Recurrent Neural Network to forecast daily mean temperatures. By training on a historical time-series dataset of weather conditions in Delhi, the model learns to predict the next day's mean temperature.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [1. Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
  - [2. Data Preprocessing for Time-Series](#2-data-preprocessing-for-time-series)
  - [3. RNN Model Architecture](#3-rnn-model-architecture)
  - [4. Model Training and Evaluation](#4-model-training-and-evaluation)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Installation and Usage](#installation-and-usage)

## Project Overview
Accurate weather forecasting is a classic time-series problem with significant real-world applications. This project tackles this challenge using a deep learning approach. An LSTM network, a special kind of RNN, is employed to capture temporal dependencies in historical weather data and make accurate future predictions. The model is built from scratch using TensorFlow and Keras.

## Dataset
The project utilizes the "Daily Delhi Climate" dataset (`DailyDelhiClimateTrain.csv`), which contains four primary weather features recorded daily from 2013 to 2017:
- `meantemp`: Mean temperature for the day
- `humidity`: Mean humidity for the day
- `wind_speed`: Mean wind speed for the day
- `meanpressure`: Mean pressure for the day

The `meantemp` feature is used as the target variable for forecasting.

## Methodology

### 1. Exploratory Data Analysis (EDA)
The first step was to understand the dataset's characteristics.
- **Data Cleaning:** The dataset was checked for null values, which were dropped to ensure data quality. The `date` column was converted to a datetime format and set as the index.
- **Time-Series Visualization:** Each weather feature was plotted over time to visualize trends, seasonality, and other patterns inherent in the data.

### 2. Data Preprocessing for Time-Series
To prepare the data for the LSTM model, the following steps were taken:
- **Feature Scaling:** The data was scaled to a range of [0, 1] using `MinMaxScaler`. This is a crucial step for neural networks as it helps the model converge faster and perform more reliably.
- **Sequence Generation:** The time-series data was transformed into input-output sequences. A `time_step` of 10 was chosen, meaning the model uses data from the past 10 days to predict the mean temperature for the 11th day.

### 3. RNN Model Architecture
A Sequential model was constructed using TensorFlow/Keras with the following layers:
- **LSTM Layer:** An LSTM layer with 50 units was used as the core of the network to learn long-term dependencies from the input sequences.
- **Dropout Layer:** A Dropout layer with a rate of 0.2 was added to prevent overfitting by randomly setting a fraction of input units to 0 during training.
- **Dense Output Layer:** A single neuron Dense layer was used for the final output, as the goal is to predict a single continuous value (the mean temperature).

The model was compiled using the **Adam optimizer** and **Mean Squared Error (MSE)** as the loss function, which are standard choices for regression tasks.

### 4. Model Training and Evaluation
The model was trained on the prepared sequences for 100 epochs. Its performance was evaluated on a separate validation set using two key metrics:
- **Root Mean Squared Error (RMSE):** Measures the average magnitude of the errors between predicted and actual values.
- **R-squared (R2) Score:** Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. A higher R2 score indicates a better fit.

## Results
The trained LSTM model demonstrated strong predictive performance on the validation data.
- The plot of **Actual vs. Predicted Temperatures** shows that the model's predictions closely follow the actual temperature fluctuations, indicating a high degree of accuracy.
- The low RMSE and high R-squared values further confirm the model's effectiveness in forecasting daily weather.
- The training and validation loss curves showed good convergence without significant overfitting.

## Technologies Used
- Python 3.x
- Pandas & NumPy
- Scikit-learn (for `MinMaxScaler` and performance metrics)
- TensorFlow & Keras (for building and training the RNN)
- Matplotlib & Seaborn (for data visualization)
- Jupyter Notebook

## Installation and Usage
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/weather-forecasting-rnn.git](https://github.com/your-username/weather-forecasting-rnn.git)
    cd weather-forecasting-rnn
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
    ```

3.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook "Daily_Weather_With_RNN.ipynb"
    ```