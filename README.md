# Manufacturing Anomaly Detection

A machine learning-powered system for detecting anomalies in pump sounds using advanced classification and deep learning techniques.
---

## 🔍 Overview

This project implements a comprehensive anomaly detection system for industrial pumps using:
- **Autoencoder** for unsupervised learning
- **XGBoost** with Bayesian optimization for production-grade predictions
- **Streamlit** UI for real-time inference and visualization

---

## ✨ Features

- 🎵 **Audio Feature Extraction** - Wavelet and spectral analysis
- 🤖 **Multiple ML Models** - Ensemble approach with autoencoder, DNN, and XGBoost
- 📊 **Similarity-Based Explainability** - Z-score deviations for interpretability
- 🎯 **High Performance** - 96.70% recall with cost-sensitive learning
- 📈 **Confusion Matrix Visualization** - Detailed performance metrics
- 🖥️ **Web UI** - Interactive Streamlit dashboard for predictions

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/chanakyakanakam/Manufacturing_Anamoly_Detection.git
cd Manufacturing_Anamoly_Detection
```

### 2. Create and Activate Conda Environment

```bash
conda create -n sound_cls python=3.10
conda activate sound_cls
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 📝 Usage



### Run Both (API + UI) Simultaneously

**Terminal 1 - API:**

```bash
cd api
conda activate sound_cls
python main.py
```

**Terminal 2 - UI:**

```bash
cd ui
conda activate sound_cls
streamlit run app.py
```

---

## 🛠️ Technologies

### Machine Learning
- **TensorFlow/Keras** - Deep learning models
- **XGBoost** - Gradient boosting for production
- **Scikit-learn** - Preprocessing and metrics
- **Optuna** - Bayesian hyperparameter optimization

### Audio Processing
- **Librosa** - Audio feature extraction
- **SciPy** - Signal processing

### Backend
- **FastAPI** - High-performance API framework
- **Uvicorn** - ASGI server

### Frontend
- **Streamlit** - Interactive web dashboard
- **Plotly** - Advanced visualizations
- **Pandas** - Data manipulation

### Data Management
- **NumPy** - Numerical computing
- **Pandas** - Data analysis

---

## 👤 Author

**Chanakya Kanakam**
- GitHub: [@chanakyakanakam](https://github.com/chanakyakanakam)
- Organization: IQuest Solutions Corp

---



---

**Last Updated:** April 2026
