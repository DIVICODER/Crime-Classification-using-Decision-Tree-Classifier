# Crime Classification using Decision Tree Classifier

## Description
This project aims to classify different types of crimes based on geographical location using a Decision Tree Classifier. The dataset is preprocessed to remove null values and unnecessary columns, and then the classifier is trained and tested to predict the crime type.

## Dataset
The dataset used in this project contains information about various crimes, including:
- Crime ID
- Month
- Reported by
- Falls within
- Longitude
- Latitude
- Location
- Crime type

## Data Preprocessing
- Removed columns: 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 10'
- Dropped rows with null values in the 'Crime ID' column

## Model
A Decision Tree Classifier is used for classification. The dataset is split into training and testing sets with a 60-40 ratio. The model's performance is evaluated using accuracy and a detailed classification report.

## Results
The model achieved an accuracy of approximately 34%. Below is the detailed classification report:

