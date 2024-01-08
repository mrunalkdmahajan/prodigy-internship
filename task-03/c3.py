import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from zipfile import ZipFile

# Get the current directory
current_directory = os.getcwd()

# Construct the path to the ZIP file using a raw string
zip_file_path = os.path.join(current_directory, r"task-03\bank.zip")

# Open the ZIP file and list the contents
with ZipFile(zip_file_path, 'r') as zip_ref:
    zip_contents = zip_ref.namelist()
    print("Contents of the ZIP file:", zip_contents)

    # Assuming the relevant CSV file is named "bank-full.csv"
    csv_file_name = "bank-full.csv"

    # Load the CSV file from the ZIP archive
    with zip_ref.open(csv_file_name) as file:
        # Load the dataset
        df = pd.read_csv(file, sep=';')

# Assume 'y' is the target variable indicating whether the customer will purchase ('yes' or 'no')
# Define features (X) and target variable (y)
X = df.drop('y', axis=1)
y = df['y']

# Perform one-hot encoding for categorical variables
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier with max_depth set to 4
model = DecisionTreeClassifier(random_state=42, max_depth=3)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Visualizing the decision tree
plt.figure(figsize=(15, 10))
plot_tree(model, feature_names=X.columns, class_names=['No Purchase', 'Purchase'], filled=True, rounded=True)
plt.savefig('./task-03/decision_tree.png')

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_report_result)
