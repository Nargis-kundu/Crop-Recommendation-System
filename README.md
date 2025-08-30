# 🌾 Crop Recommendation System

## Project Overview
The *Crop Recommendation System* is a machine learning–based project that helps farmers choose the most suitable crop to grow based on soil and environmental conditions.  
This system considers key parameters such as *Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, and Rainfall* to recommend the best crop.  

A *Flask web application* is built on top of the model with user registration and login features, so that different users can securely access recommendations.


## Objectives
- To recommend the most suitable crop based on soil and climate conditions  
- To build an interactive web application for farmers  
- To allow multiple users via authentication (registration & login system)  
- To provide a scalable ML + web solution for agriculture decision-making  

##  Dataset
- The dataset contains soil and weather parameters:  
  - N (Nitrogen content)  
  - P (Phosphorus content)  
  - K (Potassium content)  
  - Temperature  
  - Humidity  
  - pH  
  - Rainfall  
- Target column: *Crop label (recommended crop)*  

## Tech Stack
- *Python (Pandas, NumPy, Scikit-learn)* – Data preprocessing & ML model  
- *Flask* – Web application framework  
- *HTML, CSS* – Frontend for UI  
- *MySQL* – For storing user login/registration data  

## Machine Learning Model
- Model trained using *Random Forest Classifier* (high accuracy for classification)  
- Evaluation metrics: Accuracy, Precision, Recall  
- Input: [N, P, K, Temperature, Humidity, pH, Rainfall]  
- Output: Recommended crop (e.g., Rice, Maize, Mango, Apple, etc.)  

## Features
- User *registration & login* system  
- User-friendly *web form* to input soil & weather conditions  
- Real-time *ML-based crop prediction*  
- Simple and clean *dashboard* for results  

