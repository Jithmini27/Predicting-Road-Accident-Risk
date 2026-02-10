# 🚦 Road Accident Risk Prediction System  
### Kaggle Playground Series S5E10 | CIS6005 Computational Intelligence Mini Project

---

## 📌 Project Overview

This project was developed as part of the **CIS6005 Computational Intelligence** module (2024–2025 Semester 2).  
The aim of this mini project is to build a complete machine learning and deep learning solution for predicting the **likelihood of road accidents** based on road and environmental conditions.

The solution is based on the Kaggle competition:

🔗 https://www.kaggle.com/competitions/playground-series-s5e10

---

## 🎯 Goal

Predict the target variable:

- **accident_risk** (continuous value between 0 and 1)

This is a **regression problem**, evaluated using **RMSE (Root Mean Squared Error)**.

---

## 📂 Dataset

The dataset contains structured tabular features such as:

- Road type  
- Number of lanes  
- Curvature  
- Speed limit  
- Weather and lighting conditions  
- Accident history indicators  

Files used:

- `train.csv` → training dataset with labels  
- `test.csv` → unseen dataset for Kaggle predictions  
- `sample_submission.csv` → required submission format  

---

## ⚙️ Models Implemented

Five models were trained, evaluated, and submitted to Kaggle:

| Model | Technique | Purpose |
|------|----------|---------|
| Linear Regression | Baseline regression | Initial benchmark |
| Random Forest | Ensemble learning | Improved performance |
| LightGBM | Gradient boosting | Best single model |
| Neural Network | Deep Learning | Non-linear pattern learning |
| Blended Ensemble | Combined prediction | Final stability improvement |

---

## 📊 Evaluation Metrics

Models were compared using:

- RMSE (competition metric)
- MAE (Mean Absolute Error)
- R² Score (explained variance)

Cross-validation was also applied to ensure reliable model performance.

---

## 🖥️ Streamlit Web Application

The final trained model was deployed using **Streamlit**, allowing users to input road conditions and receive an accident risk prediction instantly.

App features:

✅ User-friendly interface  
✅ Real-time prediction  
✅ Backend ML pipeline integration  
✅ Practical deployment-ready system  

---

## 📁 Project Structure

```bash
Predicting Road Accident Risk/
│
├── artifacts/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── models/
│   ├── submission_lr.csv
│   ├── submission_rf.csv
│   ├── submission_lgbm.csv
│   ├── submission_nn.csv
│   ├── submission_blend.csv
│   ├── final_submission_best.csv
│   └── best_model.pkl
│
├── website/
│   ├── app.py
│   └── requirements.txt
│
├── notebooks/
│   └── S5E10_RoadAccidentRisk.ipynb
│
└── README.md
