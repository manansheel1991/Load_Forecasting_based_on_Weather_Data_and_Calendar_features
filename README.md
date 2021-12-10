# Load_Forecasting
Forecasting load based on Weather Data and features extracted from Calendar, using - 

1. Feed Forward Neural Networks.
2. Other algorithms to be trained very soon.

## Data
'Actuals.xlsx' file from IEEE Load Forecasting Competition. Contains Weather data (features like Temperature, Pressure, Humidity, Wind Speed, etc.) for hourly timestamps of past few years. Also contains electric load for the same time. 

## Files
'Load Forecasting using weather data and extracted calendar features.ipynb' extracts Calendar features from the timestamps. Features like Day of Week, Hour of Day, Whether it is a holiday, etc. and also does some analysis to obtain the importance of these Calendar features in predicting load.

'Proper Load Forecasting Model' file takes input from the file that extracts Calendar features, and trains a FFNN model for load forecasting.
