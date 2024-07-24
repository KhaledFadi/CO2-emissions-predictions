# CO2 Emission Prediction Using LSTM Neural Networks
## Introduction
This project aims to predict CO2 emissions using Long Short-Term Memory (LSTM) neural networks. The model is trained on historical CO2 emission data to forecast future emissions, showcasing the power of machine learning in tackling environmental challenges.
## Project Structure
* `CO2 emissions predictions.ipynb`: Jupyter Notebook containing the step-by-step implementation of the project.

* `train.csv`: Training dataset.

* `test.csv`: Testing dataset.

## Installation
To run this project, you need to have the following dependencies installed:

* Python 3.x

* Pandas

* NumPy

* scikit-learn

* TensorFlow

* Matplotlib

### You can install the required packages using pip:
pip install pandas numpy scikit-learn tensorflow matplotlib

## Data Preprocessing
The data preprocessing steps include:

* Loading the training and testing datasets.

* Dropping unnecessary columns and handling missing values.

* Scaling the features and target variable using MinMaxScaler.

* Creating sequences for the LSTM model using TimeseriesGenerator.

## Model Training
The LSTM model is defined using TensorFlow's Keras API. The architecture includes:

* An LSTM layer with 50 units and ReLU activation.
  
* A Dense layer with a single unit.
  
* The model is compiled with the Adam optimizer and Mean Squared Error (MSE) loss function.
  
* The model is trained for 50 epochs, and the training and validation loss are plotted to evaluate the model's performance.

## Predictions
The trained model is used to make predictions on the test dataset. The predictions are scaled back to the original range and aligned with the test data indices. The actual vs. predicted emissions for the year 2021 and 2022 are plotted for comparison.

## Results
The project demonstrates the capability of LSTM neural networks in forecasting CO2 emissions. The scatter plot of actual vs. predicted emissions provides a visual representation of the model's performance.

## Conclusion
This project highlights the application of machine learning in environmental science, specifically in predicting CO2 emissions. The LSTM model offers a robust approach to time series forecasting, providing valuable insights for tackling climate change.

## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
