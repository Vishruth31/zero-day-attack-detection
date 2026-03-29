#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import helper

# ===============================
# 🔥 OUTPUT FOLDER SETUP
# ===============================
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# 🔥 GLOBAL STORAGE
results_summary = []

# ===============================
# 🔥 CREATE SEQUENCES
# ===============================
def create_sequences(data, seq_len=10):
    sequences = []
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i+seq_len])
    return np.array(sequences)

# ===============================
# 🔥 MODEL
# ===============================
def build_lstm_autoencoder(timesteps, features):

    inputs = Input(shape=(timesteps, features))

    x = LSTM(32, activation='relu', return_sequences=True)(inputs)
    x = LSTM(16, activation='relu', return_sequences=False)(x)

    x = RepeatVector(timesteps)(x)

    x = LSTM(16, activation='relu', return_sequences=True)(x)
    x = LSTM(32, activation='relu', return_sequences=True)(x)

    outputs = TimeDistributed(Dense(features))(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')

    return model

# ===============================
# 🔥 MSE
# ===============================
def compute_mse(model, X):
    if X is None or len(X) == 0:
        return np.array([])
    recon = model.predict(X, verbose=0)
    mse = np.mean(np.square(X - recon), axis=(1, 2))
    return mse

# ===============================
# 🔥 EVALUATE
# ===============================
def evaluate(model, X, name, threshold, output_file):

    global results_summary

    if X is None or len(X) == 0:
        print(f"\n{name} → Skipped (Not enough data)")
        return

    mse = compute_mse(model, X)

    anomaly = mse > threshold
    anomaly_ratio = np.mean(anomaly)

    print(f"\n{name}")
    print("Avg MSE:", np.mean(mse))
    print("Anomaly %:", anomaly_ratio)

    # 🔥 STORE RESULT
    results_summary.append((name, anomaly_ratio))

    with open(output_file, "a") as f:
        f.write(f"{name}, Avg MSE: {np.mean(mse)}, Anomaly %: {anomaly_ratio}\n")

# ===============================
# 🔥 MAIN
# ===============================
if __name__ == "__main__":

    columns_to_drop = ['ip_src', 'ip_dst']

    normal_path = 'DataFiles/CIC/biflow_Monday-WorkingHours_Fixed.csv'
    attack_folder = 'DataFiles/CIC/'

    # 🔥 SAVE CSV IN OUTPUT FOLDER
    output_file = os.path.join(
        output_dir,
        datetime.now().strftime("%d_%m_%Y__%H_%M_") + "Results_LSTM.csv"
    )

    # ===============================
    # LOAD DATA
    # ===============================
    normal = pd.read_csv(normal_path)
    normal = normal.dropna()

    normal.drop(columns_to_drop, axis=1, inplace=True)
    normal.drop(normal.columns[0], axis=1, inplace=True)

    normal, to_drop = helper.dataframe_drop_correlated_columns(normal, 0.9)

    # ===============================
    # SCALE
    # ===============================
    scaler = MinMaxScaler()
    normal_scaled = scaler.fit_transform(normal.values)

    normal_scaled = normal_scaled[:5000]

    print("Features:", normal_scaled.shape)

    # ===============================
    # CREATE SEQUENCES
    # ===============================
    sequences = create_sequences(normal_scaled, seq_len=10)

    if len(sequences) == 0:
        print("❌ Not enough data")
        exit()

    train_X, valid_X = train_test_split(sequences, test_size=0.2, random_state=1)

    # ===============================
    # MODEL
    # ===============================
    model = build_lstm_autoencoder(
        timesteps=train_X.shape[1],
        features=train_X.shape[2]
    )

    model.summary()

    # ===============================
    # TRAIN
    # ===============================
    early_stop = EarlyStopping(monitor='val_loss', patience=2)

    history = model.fit(
        train_X,
        train_X,
        epochs=5,
        batch_size=128,
        validation_data=(valid_X, valid_X),
        callbacks=[early_stop],
        verbose=1
    )

    # ===============================
    # 📊 GRAPH 1: LOSS
    # ===============================
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()

    # ===============================
    # THRESHOLD
    # ===============================
    train_mse = compute_mse(model, train_X)
    threshold = np.mean(train_mse) + 3 * np.std(train_mse)

    print("\n🔥 Threshold:", threshold)

    with open(output_file, "w") as f:
        f.write(f"Threshold: {threshold}\n")

    # ===============================
    # 📊 GRAPH 2: MSE
    # ===============================
    plt.figure()
    plt.hist(train_mse, bins=50)
    plt.axvline(threshold, linestyle='dashed', linewidth=2)
    plt.xlabel('MSE')
    plt.ylabel('Frequency')
    plt.title('MSE Distribution')
    plt.savefig(os.path.join(output_dir, 'mse_distribution.png'))
    plt.close()

    # ===============================
    # EVALUATE NORMAL
    # ===============================
    evaluate(model, train_X, "TRAIN", threshold, output_file)
    evaluate(model, valid_X, "VALIDATION", threshold, output_file)

    # ===============================
    # 📊 GRAPH 3: COMPARISON
    # ===============================
    valid_mse = compute_mse(model, valid_X)

    plt.figure()
    plt.hist(train_mse, bins=50, alpha=0.6, label='Train')
    plt.hist(valid_mse, bins=50, alpha=0.6, label='Validation')
    plt.axvline(threshold, linestyle='dashed', linewidth=2, label='Threshold')
    plt.xlabel('MSE')
    plt.ylabel('Frequency')
    plt.title('Normal vs Threshold')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'comparison.png'))
    plt.close()

    # ===============================
    # TEST ATTACK FILES
    # ===============================
    for attack_file in os.listdir(attack_folder):

        path = os.path.join(attack_folder, attack_file)

        try:
            attack = pd.read_csv(path)
            attack = attack.dropna()

            attack.drop(columns_to_drop, axis=1, inplace=True)
            attack.drop(attack.columns[0], axis=1, inplace=True)
            attack.drop(to_drop, axis=1, inplace=True)

            attack_scaled = scaler.transform(attack.values)

            attack_scaled = attack_scaled[:2000]

            if len(attack_scaled) < 10:
                print(f"{attack_file} → Skipped (too small)")
                continue

            attack_seq = create_sequences(attack_scaled, seq_len=10)

            evaluate(model, attack_seq, attack_file, threshold, output_file)

        except Exception as e:
            print(f"{attack_file} → Error: {e}")

    # ===============================
    # 📊 FINAL SUMMARY GRAPH
    # ===============================
    names = [x[0] for x in results_summary]
    values = [x[1] for x in results_summary]

    colors = ['red' if v > 0.1 else 'green' for v in values]

    plt.figure(figsize=(12,6))
    plt.bar(names, values, color=colors)

    plt.xticks(rotation=90)
    plt.xlabel("Files")
    plt.ylabel("Anomaly %")
    plt.title("Anomaly Detection Results per File")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_summary.png'))
    plt.show()