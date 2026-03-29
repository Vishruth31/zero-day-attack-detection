# 🚀 Zero-Day Attack Detection using LSTM Autoencoder

## 📌 Overview
This project implements an anomaly-based intrusion detection system using an LSTM Autoencoder to detect zero-day attacks in network traffic.

Instead of relying on known attack signatures, the model learns normal behavior and flags any deviation as suspicious.

---

## 🧠 How It Works
1. Train on normal network traffic  
2. Learn patterns using an LSTM Autoencoder  
3. Reconstruct input and calculate reconstruction error (MSE)  
4. Define a threshold (mean + 3×std)  
5. Detect anomalies:
   - Below threshold → Normal  
   - Above threshold → Attack  

---

## 📂 Project Structure
```
zero-day-attack-detection/
│
├── main.py
├── helper.py
├── autoencoder.py
├── oneclass_svm.py
├── DataFiles/
├── outputs/
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```
git clone https://github.com/Vishruth31/zero-day-attack-detection.git
cd zero-day-attack-detection
```

---

### 2. Create Virtual Environment

#### macOS / Linux
```
python3 -m venv venv
source venv/bin/activate
```

#### Windows
```
python -m venv venv
venv\Scripts\activate
```

---

### 3. Install Dependencies
```
pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib tensorflow
```

---

### 4. Prepare Dataset
Place your CSV files inside:
```
DataFiles/CIC/
```

Make sure:
- Same format as training data  
- Numeric values only  
- No missing important columns  

---

### 5. Run the Project
```
python3 main.py
```

---

## 📊 Output
After running, a folder is automatically created:
```
outputs/
```

It contains:
- loss_plot.png → Training vs validation loss  
- mse_distribution.png → Error distribution  
- comparison.png → Normal vs threshold  
- final_summary.png → File-wise anomaly detection  
- Results_LSTM.csv → Numerical results  

---

## 📈 Sample Results

| Attack Type | Detection |
|------------|----------|
| DDoS       | 100% |
| GoldenEye  | ~99% |
| Hulk       | 100% |
| Normal     | Low anomaly |

---

## 🧪 Testing New Data
1. Add your CSV file to:
```
DataFiles/CIC/
```

2. Run:
```
python3 main.py
```

The model will automatically evaluate all files.

---

## ⚠️ Important Notes
- The model detects anomalies, not specific attack types  
- Works best with similar data as training dataset  
- Some subtle attacks may appear normal  

---

## 🎯 Key Features
- LSTM Autoencoder  
- Unsupervised anomaly detection  
- Threshold-based classification  
- Visualization graphs  
- Organized output folder  

---

## 🧠 Concept Summary
The model learns normal network behavior and flags deviations using reconstruction error.

---

## 📌 Future Improvements
- Real-time detection  
- Hybrid models  
- GUI dashboard  
- Cloud deployment  

---

## 👨‍💻 Author
Vishruth  
https://github.com/Vishruth31  

---

## ⭐ Support
If you found this useful, give it a star ⭐
