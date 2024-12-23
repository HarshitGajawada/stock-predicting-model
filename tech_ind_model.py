import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(4)
tf.random.set_seed(4)
from util import csv_to_dataset, calc_ema, history_points


# dataset

ohlcv_histories, technical_indicators, next_day_open_values, y_normaliser = csv_to_dataset('ZOMATO.BSE_daily.csv')

# Model architecture
lstm_input = Input(shape=(history_points, 5), name='lstm_input')
dense_input = Input(shape=(technical_indicators.shape[1],), name='tech_input')

# LSTM branch
x = LSTM(50, name='lstm_0')(lstm_input)
x = Dropout(0.2, name='lstm_dropout_0')(x)
lstm_branch = Model(inputs=lstm_input, outputs=x)

# Technical indicators branch
y = Dense(20, name='tech_dense_0')(dense_input)
y = Activation("relu", name='tech_relu_0')(y)
y = Dropout(0.2, name='tech_dropout_0')(y)
technical_indicators_branch = Model(inputs=dense_input, outputs=y)

# Combine branches
combined = concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate')

# Final dense layers
z = Dense(64, activation="sigmoid", name='dense_pooling')(combined)
z = Dense(1, activation="linear", name='dense_out')(z)

# Build the model
model = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)
adam = optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=adam, loss='mse')

# Train the model
model.fit(
    x=[ohlcv_histories, technical_indicators],
    y=next_day_open_values,
    batch_size=32,
    epochs=50,
    shuffle=True,
    validation_split=0.1
)

# Predict the next 30 days
last_ohlcv = ohlcv_histories[-1]
last_tech_ind = np.array([np.mean(last_ohlcv[:, 3])])  # Example with a single feature
future_predictions = []

for _ in range(30):
    pred = model.predict([np.expand_dims(last_ohlcv, axis=0), np.expand_dims(last_tech_ind, axis=0)])
    future_predictions.append(pred[0, 0])
    # Update last_ohlcv for next prediction
    last_ohlcv = np.roll(last_ohlcv, shift=-1, axis=0)
    last_ohlcv[-1, 0] = pred[0, 0]  # Replace with predicted open price
    last_tech_ind = np.array([np.mean(last_ohlcv[:, 3])])  # Example with a single feature

# Rescale predictions
future_predictions = y_normaliser.inverse_transform(np.array(future_predictions).reshape(-1, 1))
past_data = y_normaliser.inverse_transform(next_day_open_values.reshape(-1, 1))
past_data = past_data[-30:]

# Plot the predictions
plt.figure(figsize=(15, 7))
plt.plot(past_data, label='Past Data', color='blue')
plt.plot(range(len(past_data), len(past_data) + len(future_predictions)), future_predictions, label='Future Predictions (Next 30 Days)', color='red')
plt.legend()
plt.show()
