import pandas as pd
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv(r"crime_classification.csv")

# Display the first few rows of the dataset
print(df.head())

# Check the shape of the dataset
print(df.shape)

# Check for null values
print(df.isnull().sum())

# Drop unnecessary columns
data = df.drop(columns=['Unnamed: 7', 'Unnamed: 8', 'Unnamed: 10'])

# Check for null values after dropping columns
print(data.isnull().sum())

# Drop rows with null values in the 'Crime ID' column
data_cleaned = data.dropna(subset=['Crime ID'])

# Check for null values in the cleaned data
print(data_cleaned.isnull().sum())

# Prepare the feature and target variables
x = data_cleaned[['Latitude', 'Longitude']]
y = data_cleaned['Crime type']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

# Initialize the Decision Tree Classifier
model = DecisionTreeClassifier()

# Train the model
model.fit(x_train, y_train)

# Make predictions
predictions = model.predict(x_test)

# Evaluate the model
score = accuracy_score(y_test, predictions)
result = classification_report(y_test, predictions)

# Print the accuracy score and classification report
print("Accuracy:", score)
print("Classification Report:\n", result)
