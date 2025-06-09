# Importing necessary libraries
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For visualization
import pandas as pd  # For data manipulation

# Loading the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# Splitting dataset into features (X) and target variable (y)
X = dataset.iloc[:, :-1].values  # All columns except the last (Age and Estimated Salary)
y = dataset.iloc[:, -1].values   # The last column (Purchased: 0 or 1)

# Splitting the dataset into Training and Test sets
from sklearn.model_selection import train_test_split
# 75% for training, 25% for testing; random_state ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Output the training and testing data (optional for understanding structure)
print(X_train)
print(y_train)
print(X_test)
print(y_test)

# Feature Scaling: standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  # Fit to training data and transform it
X_test = sc.transform(X_test)        # Only transform the test data using the same scaling

# View the scaled features (optional)
print(X_train)
print(X_test)

# Training the Logistic Regression model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)  # Create the classifier object
classifier.fit(X_train, y_train)  # Fit model to the training data

# Predicting a single new result (e.g., person aged 30 with $87,000 salary)
# The input is scaled using the same StandardScaler
print(classifier.predict(sc.transform([[30,87000]])))  # Output will be 0 or 1

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Comparing predicted results and actual test labels side by side
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Evaluating model performance with confusion matrix and accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)  # Shows TP, TN, FP, FN
print(cm)
accuracy_score(y_test, y_pred)  # Prints accuracy of the model

# Visualizing the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train  # Convert scaled data back to original
X1, X2 = np.meshgrid(
    np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
    np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25)
)

# Predicting the output (0 or 1) for each point on the grid
plt.contourf(X1, X2,
             classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(['#FA8072', '#1E90FF']))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# Plotting actual training points with colors based on class label
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(['#FA8072', '#1E90FF'])(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualizing the Test set results (same as training visualization but with test data)
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.25),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.25)
)
Z = classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)
plt.contourf(X1, X2, Z, alpha=0.75, cmap = ListedColormap(['#FA8072', '#1E90FF']))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

colors = ['#FA8072', '#1E90FF']
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], color=colors[i], label=j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
