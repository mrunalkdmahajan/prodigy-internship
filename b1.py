#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:\\Users\\mruna\\Internship\\task-02\\train.csv')
df.head()

df.shape

#removing the columns
df = df.drop(columns=['PassengerId','Name','Cabin','Ticket'], axis= 1)
print(df.head())

print(df.describe())

#checking data types
df.dtypes
#checking for unique value count
df.nunique()

#checking for missing value count
df.isnull().sum()

# replacing the missing values
df['Age'] =  df['Age'].replace(np.nan,df['Age'].median(axis=0))
df['Embarked'] = df['Embarked'].replace(np.nan, 'S')

# Type casting 'Age' to integer
df['Age'] = df['Age'].astype(int)

# Replacing 'male' with 1 and 'female' with 0 in the 'Sex' column
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
# Creating age groups
bins = [0, 5, 20, 30, 40, 50, 60, 100]
labels = ['Infant', 'Teen', '20s', '30s', '40s', '50s', 'Elder']
df['Age'] = pd.cut(df['Age'], bins=bins, labels=labels)

#
# Exploratory Data Analysis

# Visualizing the count of features
fig, ax = plt.subplots(2, 4, figsize=(20, 20))

sns.countplot(x='Survived', data=df, ax=ax[0, 0])
sns.countplot(x='Pclass', data=df, ax=ax[0, 1])
sns.countplot(x='Sex', data=df, ax=ax[0, 2])
sns.countplot(x='Age', data=df, ax=ax[0, 3])
sns.countplot(x='Embarked', data=df, ax=ax[1, 0])
sns.histplot(x='Fare', data=df, bins=10, ax=ax[1, 1])
sns.countplot(x='SibSp', data=df, ax=ax[1, 2])
sns.countplot(x='Parch', data=df, ax=ax[1, 3])

# Visualizing the count of features with respect to 'Survived'
fig, ax = plt.subplots(2, 4, figsize=(20, 20))

sns.countplot(x='Survived', data=df, ax=ax[0, 0])
sns.countplot(x='Pclass', data=df, hue='Survived', ax=ax[0, 1])
sns.countplot(x='Sex', data=df, hue='Survived', ax=ax[0, 2])
sns.countplot(x='Age', data=df, hue='Survived', ax=ax[0, 3])
sns.countplot(x='Embarked', data=df, hue='Survived', ax=ax[1, 0])
sns.histplot(x='Fare', data=df, bins=10, hue='Survived', ax=ax[1, 1])
sns.countplot(x='SibSp', data=df, hue='Survived', ax=ax[1, 2])
sns.countplot(x='Parch', data=df, hue='Survived', ax=ax[1, 3])

plt.show()

