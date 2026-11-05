<div align="center">

# 🚀 Predicting Startup Acquisition

**Machine Learning Pipeline & Flask Web Application for Startup Outcome Prediction**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

<br>

## 📖 Overview

This repository contains an end-to-end **Machine Learning Pipeline and Flask Web Application** designed to predict startup outcomes using historical startup data.

The project combines data preprocessing, feature engineering, machine learning model training, evaluation, and a web-based prediction interface.

A **Random Forest Classifier** is used to perform multiclass startup outcome prediction based on financial, operational, geographical, industry, and historical startup features.

Users can enter startup information through the web interface and receive a predicted outcome along with prediction probabilities and confidence.

---

## 🎯 Project Objectives

- Build an end-to-end machine learning pipeline for startup outcome prediction.
- Perform preprocessing and feature engineering on startup data.
- Handle categorical startup attributes using encoded features.
- Train and evaluate a Random Forest classification model.
- Handle class imbalance using balanced class weights.
- Deploy the trained model through a Flask web application.
- Provide a simple interface for users to make predictions.
- Expose a REST API for programmatic predictions.

---

## ✨ Key Features

- **🧠 Random Forest Multiclass Classification**
- **🎯 4-Class Startup Outcome Prediction**
- **🔢 34 Model Features**
- **🌐 Interactive Flask Web Interface**
- **✅ Input Validation**
- **📊 Prediction Probabilities**
- **📈 Prediction Confidence**
- **🔌 REST API**
- **🩺 Model Health Check Endpoint**
- **🧪 Sample Data Testing**
- **⚖️ Class-Balanced Model Training**

---

## 🤖 Machine Learning Model

### Random Forest Classifier

The final application uses a Random Forest classifier with balanced class weights to improve learning across imbalanced startup outcome classes.

### Model Configuration

```text
Algorithm          : Random Forest
Estimators         : 100
Maximum Depth      : 15
Minimum Samples Leaf : 2
Class Weight       : Balanced
Parallel Jobs      : 2
## Project Attribution

Original project author: saivyshnavi30
Repository owner and maintainer: Guguloth-Devaharshini
