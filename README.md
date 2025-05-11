# 🍽️ Food Analysis Web Application

This project is a **Food Analysis Web App** that helps users analyze nutritional content using the **USDA Food API** and predict food-related outcomes using **Machine Learning models** like **XGBoost**, **Random Forest**, and **SVM**. The application also features a clean web interface with a **history function** backed by **Firebase** for user data tracking.

## 🚀 Features

- 🔍 Search foods and retrieve nutrition data from the USDA API.
- 🤖 Predictive analysis using trained ML models (XGBoost, Random Forest, SVM).
- 📊 Display nutritional facts and prediction results in a user-friendly format.
- 🕓 Save and view history of past searches using Firebase Firestore.
- 🌐 Intuitive web interface for easy user interaction.

## 🛠️ Tech Stack

- **Frontend**: HTML, CSS, JavaScript (or specify React/Flask template if used)
- **Backend**: Python (Flask or FastAPI)
- **APIs**: USDA FoodData Central API
- **Machine Learning**: XGBoost, Random Forest, SVM (scikit-learn)
- **Database**: Firebase Firestore for user history

## 🔍 How It Works

1. **User Inputs Food Name** → e.g., "apple"
2. **USDA API Call** → Retrieves nutrition facts (calories, fats, protein, etc.)
3. **ML Model Prediction** → Predicts a custom output (e.g., health score, category, etc.)
4. **Display Result** → Nutritional breakdown + prediction shown on web interface
5. **History** → Logged in Firebase with timestamp and food data
