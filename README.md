# CGM-Prediction-Models
Source code for predictive models used in Feature-Driven Deep Neural Network For Predicting Blood Glucose Levels paper.

# BiGRU and BiLSTM Models for CGM Prediction

This repository contains the implementation of deep learning models for **Continuous Glucose Monitoring (CGM) prediction** at different **Prediction Horizons (PHs)**.
The main architecture is a **Bidirectional GRU (BiGRU) and Bidirectional LSTM (BiLSTM) models**, evaluated using standard error metrics and **Prediction Error Grid Analysis (PRED-EGA)**.

The code was originally developed in Google Colab and is designed to load preprocessed datasets from Google Drive.
---

## Features

* Sliding window approach for time-series prediction.
* Configurable **prediction horizons (PH)**: 5, 30, 60 minutes.
* Deep learning architecture: BiLSTM and BiGRU models with dropout and regularization techniques.
* Evaluation metrics: **MAE, MSE, RMSE**.
* **PRED-EGA (Prediction Error Grid Analysis)** implementation for clinical evaluation.
* Visualization of training loss and prediction performance.
---


## Usage

1. **Upload Dataset**
   Place your preprocessed CSV dataset in Google Drive and update the path in the code:

   ```python
   path = 'Path_to_where_the_data_is_located_in_the_drive'
   ```

2. **Run the Notebook**
   Open the notebook in Google Colab and run all cells.

3. **Change Prediction Horizon (PH)**
   Modify the `ph` value to test different horizons:

   ```python
   ph = 5   # for 5-minute prediction
   ph = 30  # for 30-minute prediction
   ph = 60  # for 60-minute prediction
---


## License

The code is provided for academic and research purposes only. Commercial use is strictly prohibited.
