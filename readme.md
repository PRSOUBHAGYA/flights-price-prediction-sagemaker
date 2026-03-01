# ✈️ Flight Price Prediction using AWS SageMaker

This repository contains an end-to-end Machine Learning project to predict flight ticket prices in India. The project demonstrates a full MLOps lifecycle: from data cleaning and feature engineering to model training on **AWS SageMaker** and deployment via a **Streamlit** web interface.

---

## 🏗️ Architecture Overview

The system architecture is designed to be scalable and cloud-native:
1. **Data Storage:** Raw and processed datasets are stored in **Amazon S3**.
2. **Preprocessing:** Data cleaning and feature engineering (One-Hot Encoding, Scaling) are performed using `Scikit-Learn` and `feature_engine`.
3. **Model Training:** An **XGBoost** regressor is trained using SageMaker's managed training instances.
4. **Deployment:** The model is hosted as a **SageMaker Endpoint** for real-time inference.
5. **Frontend:** A **Streamlit** dashboard acts as the client, sending user inputs to the AWS endpoint via `boto3`.



---

## 📂 Project Structure

```text
├── data/                         # Train and Test CSV files
├── notebooks/                    # Step-by-step development
│   ├── 01_Data_Cleaning.ipynb     # Initial cleaning & type conversion
│   ├── 02_EDA.ipynb               # Exploratory Data Analysis
│   ├── 03_Feature_Engineering.ipynb # Pipeline creation (Encoding/Scaling)
│   └── 04_SageMaker_Training.ipynb  # AWS Model training & deployment
├── app.py                        # Streamlit web application code
├── preprocessor.joblib           # Serialized preprocessing pipeline
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation