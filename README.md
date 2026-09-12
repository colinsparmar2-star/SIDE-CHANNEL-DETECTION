# SIDE-CHANNEL-DETECTION
# Side-Channel Detection Using Machine Learning

A machine learning project for detecting network intrusions using classification algorithms.

## Project Overview

This project explores the use of machine learning techniques for detecting malicious network activity. Multiple classification models are trained and evaluated to distinguish between normal and attack traffic.

The repository contains two implementations of the project:

- *Local VS Code* — Python implementation using a generated synthetic dataset.
- *Google Colab* — Python implementation using the KDDTrain+ dataset.

## Machine Learning Models

The project uses:

- Logistic Regression
- Random Forest
- XGBoost

## Evaluation

The models are evaluated using:

- Accuracy
- Confusion Matrix
- Classification Report
- ROC Curve
- AUC

The Local VS Code implementation achieved approximately:

| Model | Accuracy |
|---|---:|
| Logistic Regression | 82.1% |
| Random Forest | 94.9% |
| XGBoost | 96.5% |

## Dataset

The Google Colab implementation uses the *KDDTrain+* dataset for network intrusion detection.

The dataset contains normal network traffic and different types of attack traffic.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Google Colab
- Visual Studio Code

## Project Structure

```text
SIDE-CHANNEL-DETECTION/
│
├── Google Colab/
│   └── Google Colab implementation
│
├── Local VS Code/
│   └── side_channel_detection.py
│
├── KDDTrain+.txt
├── README.md
├── .gitignore
└── .gitattributes
