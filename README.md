# Disease Prediction Web App

A Machine Learning-based web application that predicts the likelihood of Diabetes, Heart Disease, and Obesity using user-provided health information. The application is built with Flask and uses trained Random Forest models for disease prediction.

## Features

* Diabetes Prediction
* Heart Disease Prediction
* Obesity Prediction
* User-friendly web interface
* Machine Learning-powered predictions
* Data preprocessing and feature scaling
* Confidence score for predictions

## Technologies Used

* Python
* Flask
* Pandas
* NumPy
* Scikit-learn
* HTML
* CSS

## Project Structure

```
project/
│
├── app.py
├── train_model.py
├── model/
│   ├── diabetes_model.pkl
│   ├── diabetes_scaler.pkl
│   ├── heart_model.pkl
│   ├── heart_scaler.pkl
│   ├── obesity_model.pkl
│   └── obesity_scaler.pkl
│
├── dataset/
│   ├── diabetes.csv
│   ├── heart.csv
│   └── obesity.csv
│
├── templates/
│   ├── index.html
│   ├── diabetes.html
│   ├── heart.html
│   ├── obesity.html
│   ├── result.html
│   └── error.html
│
└── static/
```

## Installation

1. Clone the repository

```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Train the models

```bash
python train_model.py
```

4. Run the Flask application

```bash
python app.py
```

5. Open your browser and visit

```
http://127.0.0.1:5000
```

## Machine Learning Models

* Random Forest Classifier
* StandardScaler for feature normalization
* GridSearchCV for hyperparameter tuning

## Future Improvements

* Additional disease prediction modules
* Improved UI/UX
* Model performance enhancements
* Cloud deployment support

## Author

Karthikeyan S A

## License

This project is licensed under the MIT License.
