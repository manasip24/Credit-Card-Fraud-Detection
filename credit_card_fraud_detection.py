#!/Users/manasi/anaconda3/bin/python

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Specify the full path to your CSV file
file_path = '/Users/manasi/Downloads/creditcard.csv'

# Loading the dataset into a pandas DataFrame
df = pd.read_csv(file_path)

################################ Data Preprocessing ################################

# Checking missing values in the dataset
missing_values = df.isnull().sum()

# Removing rows with missing values
df.dropna(inplace=True)

# Step 3: Outlier Detection and Handling

# Visualizing potential outliers (for example, 'Amount' can be a column to check for outliers)
sns.boxplot(x=df['Amount'])
plt.title("Box Plot of 'Amount'")

# Identifying and handling outliers (e.g., winsorization)

# Winsorizing 'Amount' to limit the effect of outliers
df['Amount'] = winsorize(df['Amount'], limits=[0.05, 0.05])

# Step 4: Class Imbalance Handling

# Checking the class distribution ('Class' is the target column indicating fraud or not)
class_distribution = df['Class'].value_counts()
print("Class Distribution:\n", class_distribution)

# Implementing class imbalance handling techniques (e.g., oversampling)
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(df.drop('Class', axis=1), df['Class'])

# Checking the new class distribution after oversampling
new_class_distribution = pd.Series(y_resampled).value_counts()
print("New Class Distribution:\n", new_class_distribution)

# Step 6: Scaling

# Normalize or standardize numerical features (e.g., using StandardScaler)
scaler = StandardScaler()
df[['Amount']] = scaler.fit_transform(df[['Amount']])

# Repeat scaling for all relevant numerical features, if they exist in your dataset

# Step 7: Data Splitting

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

################################ Connecting to DB ################################


# Make sure to handle exceptions and close the connection when done with database operations.
import mysql.connector

# Establish a connection to the MySQL database
db_connection = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="Kunasi@2414",
    database="Fraud Detection"  # Replace with your actual database name
)

# Create a cursor object to interact with the database
cursor = db_connection.cursor()

import mysql.connector

# Establish a connection to the MySQL database
db_connection = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="Kunasi@2414",
    database="Fraud Detection"  # Replace with your actual database name
)

# Create a cursor object to interact with the database
cursor = db_connection.cursor()

# Example data to insert (replace with your actual preprocessed data)
data_to_insert = [
    (1, 100.00, '2023-01-01 12:00:00', 25.50, 0.75, 0),
    (2, 75.50, '2023-01-02 14:30:00', 30.25, 0.85, 1),
    # Add more rows of data as needed
]

# SQL query to insert data into the preprocessed_data table
insert_data_query = """
INSERT INTO preprocessed_data (transaction_id, preprocessed_amount, preprocessed_time, preprocessed_feature1, preprocessed_feature2, preprocessed_fraud_flag)
VALUES (%s, %s, %s, %s, %s, %s);
"""

# Execute the insert query for each row of data
for row_data in data_to_insert:
    cursor.execute(insert_data_query, row_data)

# Commit the changes to save the inserted data
db_connection.commit()

################################ Data Retrieval ################################

# verification query to (Verify that data in the preprocessed_data table is ready for modeling)
#import mysql.connector
import pandas as pd


# Define a SQL query to retrieve the data (replace with your desired query)
query = "SELECT * FROM preprocessed_data"

# Fetch the data from the database into a Pandas DataFrame
preprocessed_data = pd.read_sql(query, db_connection)

# Assuming 'preprocessed_time' is the datetime column
preprocessed_data['preprocessed_time'] = pd.to_datetime(preprocessed_data['preprocessed_time'])
preprocessed_data['year'] = preprocessed_data['preprocessed_time'].dt.year
preprocessed_data['month'] = preprocessed_data['preprocessed_time'].dt.month
preprocessed_data['day'] = preprocessed_data['preprocessed_time'].dt.day
preprocessed_data['hour'] = preprocessed_data['preprocessed_time'].dt.hour
preprocessed_data['minute'] = preprocessed_data['preprocessed_time'].dt.minute
preprocessed_data['second'] = preprocessed_data['preprocessed_time'].dt.second

# Now drop the original datetime column as we have extracted the features
preprocessed_data = preprocessed_data.drop(columns=['preprocessed_time'])


# Close the database connection when done
db_connection.close()

################################ Data Splitting ################################

from sklearn.model_selection import train_test_split

# Split the data into features (X) and the target variable (y)
X = preprocessed_data.drop(columns=['preprocessed_fraud_flag'])
y = preprocessed_data['preprocessed_fraud_flag']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

################################ Model Training ################################
from sklearn.linear_model import LogisticRegression

# Initialize the Logistic Regression model (you can specify hyperparameters here)
model = LogisticRegression()
# Fit the model to the training data
model.fit(X_train, y_train)


################################ Model Evaluation ################################
# Step 5: Model Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Predict the target variable on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"ROC-AUC Score: {roc_auc:.2f}")


################################ Model Visualization (Optional) ################################
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

