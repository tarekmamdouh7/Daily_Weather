# Temperature Time-Series Prediction with SimpleRNN

An exploratory weather-data notebook that uses the previous 10 temperature observations to predict the next observation. The model is built with TensorFlow/Keras and uses a SimpleRNN layer, not an LSTM.

## Project contents

- `Daily_Weather_With_RNN.ipynb`: weather-data exploration, preprocessing, sequence generation, model training and evaluation.

## Dataset

The notebook expects an external CSV named `Task 2 - Daily Weather Dataset.csv`. The dataset is not included in this repository.

The saved inspection shows 420,551 rows and 15 original columns. Relevant fields include:

- `Date Time`: observation timestamp.
- `T (degC)`: temperature, used for prediction.
- `rh (%)`: relative humidity, used in exploratory plots.
- Other measured weather variables such as pressure, wind speed and vapor pressure.

Although the repository is named Daily_Weather, the model operates on consecutive observations. The notebook does not resample the measurements into daily averages, so its prediction horizon should not be described as the next day.

## Workflow

1. Inspect the dataset and remove duplicate rows.
2. Parse timestamps and extract calendar features.
3. Plot temperature, humidity, monthly patterns, hourly patterns and a month-by-hour temperature heatmap.
4. Sort observations by timestamp and select the temperature series.
5. Scale temperature with MinMaxScaler.
6. Build sliding windows of 10 observations with the next temperature as the target.
7. Create an 80/20 random train/test split with `random_state=42`.
8. Train the model and inspect loss curves and error metrics.

## Model

- SimpleRNN with 50 units.
- Dropout of 0.2.
- Dense output layer with one unit.
- Adam optimizer and mean squared error loss.
- 20 training epochs with a batch size of 32.

## Evaluation

The notebook calculates MAE, RMSE, MAPE and R-squared on scaled temperature values. MAE and RMSE are therefore not reported in degrees Celsius.

Two saved evaluation cells show different results despite using the same prediction and metric logic. They should be rerun consistently before selecting a headline score.

The current random split can put overlapping sequence windows in both training and test data. The scaler is also fitted before splitting, and the test set is used as validation data during training. These choices limit the interpretation of the saved metrics as future-forecast performance.

## Run locally

1. Download or clone this repository.
2. Install the main dependencies:

   ```bash
   pip install jupyter pandas numpy scikit-learn tensorflow matplotlib seaborn
   ```

3. Place `Task 2 - Daily Weather Dataset.csv` beside the notebook, or update the CSV path.
4. Open the notebook:

   ```bash
   jupyter notebook "Daily_Weather_With_RNN.ipynb"
   ```

5. Run the cells in order to inspect the data, build the sequences and train the model. Exact package versions are not pinned.

## Recommended next steps

Use chronological training, validation and test periods; fit the scaler on training data only; compare against a last-observation baseline; and inverse-transform predictions before reporting errors in degrees Celsius. These changes are needed before drawing strong conclusions about forecasting performance.
