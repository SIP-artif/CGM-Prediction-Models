# Import Libraries

from google.colab import drive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Bidirectional, Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error

"""# Uploading the preprocessed dataset from the Drive"""

drive.mount('/to_your/drive')

path = 'Path_to_where_the_data_is_located_in_the_drive'

dataset = pd.read_csv(path)

dataset = dataset.set_index('EventDateTime')

"""# Splitting the data into:
70% Training
10% Validation
20% Testing
"""

def split_dataset_by_subject(dataset, train_size, val_size):
    """
    Splits the dataset into training, validation, and testing sets based on the subject IDs.

    Args:
        dataset (DataFrame): The input dataset with a 'Subject_ID' column.
        train_size (int): The number of subjects in the training set.
        val_size (int): The number of subjects in the validation set.

    Returns:
        tuple: A tuple containing the training, validation, and testing pandas DataFrames.
    """
    subjects = dataset['Subject_ID'].unique()

    #temp_subjects holds the validation and testing subjects temporary
    train_subjects, temp_subjects = train_test_split(
        subjects, train_size = train_size, random_state=42)

    val_subjects, test_subjects = train_test_split(
        temp_subjects, train_size = val_size, random_state=42)


    training = dataset[dataset['Subject_ID'].isin(train_subjects)].copy()
    validation   = dataset[dataset['Subject_ID'].isin(val_subjects)].copy()
    testing  = dataset[dataset['Subject_ID'].isin(test_subjects)].copy()

    return training, validation, testing

training, validation, testing = split_dataset_by_subject(dataset, train_size=17, val_size=3)

#No more need for the subject_ID column
training.drop('Subject_ID', axis=1, inplace=True)
validation.drop('Subject_ID', axis=1, inplace=True)
testing.drop('Subject_ID', axis=1, inplace=True)

"""# Window Sliding
Time step, which is the number of the step the model will take from the past to predict future.
we choose 36: **3 hours past** (3*60/5 = 36)
"""

def sliding(data, time_step, ph_minutes):
    """
    Takes a training, validation sets and generates input-output pairs (X,y) to
    create a sliding windows.

    Args:
        data (array): The input data.
        time_step (int): The number of past time steps to include in each input sequence (window size).
        ph_minutes (int): The prediction horizon in minutes, determines how far into the future the target value is located.

    Returns:
        tuple: A tuple containing two arrays:
            - X (array): The input sequences with shape (number of samples, time_step, number of features).
                              Each sample is a sequence of past data points minus the target variable.
            - y (array): The target values with shape (number of samples,).
    """

    X, y = [], []
    horizon_steps = int(ph_minutes / 5)  # The EventDateTime column have 5-min frequency

    for i in range(len(data) - time_step - horizon_steps):
        X.append(data[i:(i + time_step), :-1])
        y.append(data[i + time_step + horizon_steps - 1, -1])

    return np.array(X), np.array(y)

time_step = 36
#ph = 5
ph = 30
#ph = 60
X_train, y_train = sliding(training.values, time_step, ph)
X_valid, y_valid = sliding(validation.values, time_step, ph)
X_test, y_test = sliding(testing.values, time_step, ph)

X_train.shape

"""# Normalize the target values"""

#Scaling the target value
scaler = MinMaxScaler()

y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))
y_valid_scaled = scaler.transform(y_valid.reshape(-1, 1))
y_test_scaled = scaler.transform(y_test.reshape(-1, 1))

"""# GRU Model"""

model = Sequential()

# Input layer
model.add(GRU(128, activation='tanh', return_sequences=True, input_shape=(36, 5), kernel_regularizer=regularizers.l1_l2(l1=1e-6, l2=1e-5)))
model.add(Dropout(0.4))

#Second layer
model.add(Bidirectional(GRU(64, activation='tanh', return_sequences=True, kernel_regularizer=regularizers.l1_l2(l1=1e-6, l2=1e-5))))
model.add(Dropout(0.4))

# Third layer
model.add(Bidirectional(GRU(32, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-6, l2=1e-5))))
model.add(Dropout(0.4))

# Output layer
model.add(Dense(1, activation='linear', kernel_regularizer=regularizers.l1_l2(l1=1e-6, l2=1e-6)))

"""# Model Training"""

# EarlyStopping: Stop training when the validation loss has stopped improving.
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ReduceLROnPlateau: Reduce learning rate when the validation loss has stopped improving.
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

optimizer = Adam(learning_rate=1e-4)

model.compile(optimizer=optimizer, loss='mae', metrics=['mse'])

history = model.fit(
    X_train, y_train_scaled,
    epochs=100,
    batch_size=128,
    validation_data=(X_valid, y_valid_scaled),
    callbacks=[early_stop, lr_scheduler]
    )

"""# Evaluating"""

#Evaluating the scaled value
mae, mse= model.evaluate(X_test, y_test_scaled)
rmse = np.sqrt(mse)

print(f"Scaled MAE: {mae:.4f}")
print(f"Scaled MSE: {mse:.4f}")
print(f"Scaled RMSE: {rmse:.4f}")

y_pred_scaled = model.predict(X_test)

#Inverse the normalization
y_pred = scaler.inverse_transform(y_pred_scaled)
y_true = scaler.inverse_transform(y_test_scaled)

#Evaluating the real values
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mse)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

#Plotting the Real vs. predicted CGM values
plt.figure(figsize=(14, 4))
plt.plot(y_true, label='Real CGM')
plt.plot(y_pred, label='Predicted CGM')
plt.title('Real vs Predicted CGM Values')
plt.xlabel('Sample')
plt.ylabel('CGM')
plt.legend()
plt.grid(True)
plt.show()

"""# Clinical Evaluation (PRED-EGA)
Prediction Error Grid Analysis
"""

def pred_ega(y_true, y_pred, glycemic_ranges=(40, 400)):
    """
    Generates the PRED-EGA.

    Args:
        y_true (array): Array of actual CGM values.
        y_pred (array): Array of predicted CGM values.
        glycemic_ranges (tuple): Tuple specifying the lower and upper bounds of the
                                 normoglycemic range, default is (70, 180).

    Returns:
        tuple: A tuple containing:
            - zones (dict): The counts of data points in each zone (A, B, C, D, E).
            - fig (Figure): The figure object of the PRED-EGA plot.
            - ax (Axes): The axes object of the PRED-EGA plot.
    """

    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have the same length")

    lower_bound, upper_bound = glycemic_ranges
    zones = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
    total_points = len(y_true)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, s=5, alpha=0.5)
    ax.set_title('PRED-EGA Analysis')
    ax.set_xlabel('Actual CGM (mg/dL)')
    ax.set_ylabel('Predicted CGM (mg/dL)')
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)

    for i in range(total_points):
        true_val = y_true[i]
        pred_val = y_pred[i]

        # Zone A: predicted within +/- 20 mg/dL or within 20%
        if (true_val >= 70 and abs(pred_val - true_val) <= 0.2 * true_val) or (true_val < 70 and abs(pred_val - true_val) <= 20):
          zones['A'] += 1

        # Zone B: clinically acceptable errors
        elif (pred_val > lower_bound and pred_val < upper_bound and true_val > lower_bound and true_val < upper_bound) or \
             (true_val <= lower_bound and pred_val > true_val - 70 and pred_val < true_val + 50) or \
             (true_val >= upper_bound and pred_val < true_val + 70 and pred_val > true_val - 50):
            zones['B'] += 1

        # Zone C: errors leading to unnecessary treatment
        elif (true_val > lower_bound and true_val < upper_bound and (pred_val <= lower_bound or pred_val >= upper_bound)):
            zones['C'] += 1

        # Zone D: errors leading to dangerous treatment
        elif (true_val <= lower_bound and pred_val >= upper_bound) or (true_val >= upper_bound and pred_val <= lower_bound):
            zones['D'] += 1

        # Zone E: errors indicating prediction failure
        elif (true_val < lower_bound and pred_val > upper_bound) or (true_val > upper_bound and pred_val < lower_bound):
          zones['E'] += 1


    ax.plot([0, 400], [0, 400], 'k--')
    ax.axhline(lower_bound, color='gray', linestyle='--', linewidth=0.5)
    ax.axhline(upper_bound, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(lower_bound, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(upper_bound, color='gray', linestyle='--', linewidth=0.5)

    return zones, fig, ax

zones_pred_ega, fig_pred_ega, ax_pred_ega = pred_ega(y_true.flatten(), y_pred.flatten())

print("PRED-EGA Zones:")
for zone, count in zones_pred_ega.items():
    percentage = (count / len(y_true)) * 100
    print(f"Zone {zone}: {count} ({percentage:.2f}%)")

plt.show()

"""#Visualization of the model loss"""

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
