# Import necessary libraries
import numpy as np                         # For numerical operations
import matplotlib.pyplot as plt            # For data visualization
import pandas as pd                        # For data handling and analysis

# Load dataset
dataset = pd.read_csv('Social_Network_Ads.csv')  # Read CSV file into a DataFrame
X = dataset.iloc[:, :-1].values                 # Extract all columns except the last as features
y = dataset.iloc[:, -1].values                  # Extract the last column as target (label)

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.25, random_state = 0)    # 25% for testing, 75% for training

# Print the training and test data for inspection
print(X_train)
print(y_train)
print(X_test)
print(y_test)

# Apply feature scaling to standardize input features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)              # Fit on training set, transform it
X_test = sc.transform(X_test)                    # Use same scaler to transform test set

# Print scaled data
print(X_train)
print(X_test)

# Train the SVM model using a linear kernel
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)  # Linear kernel SVM
classifier.fit(X_train, y_train)                        # Train the model

# Predict a new example result with input values (Age=30, Salary=87000)
print(classifier.predict(sc.transform([[30, 87000]])))  # Transform input before prediction

# Predict test set results
y_pred = classifier.predict(X_test)

# Concatenate predictions and actual values for comparison
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Create a confusion matrix and calculate accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)          # Create confusion matrix
print(cm)
accuracy_score(y_test, y_pred)                 # Compute and print accuracy

# Visualize training set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train    # Reverse scaling for visualization
X1, X2 = np.meshgrid(                                     # Create coordinate grid
    np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
    np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25)
)
# Plot decision boundary and regions
plt.contourf(X1, X2, classifier.predict(
    sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
    alpha = 0.75, cmap = ListedColormap(['#FA8072', '#1E90FF'])
)
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
# Plot actual training points
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(['#FA8072', '#1E90FF'])(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualize test set results
X_set, y_set = sc.inverse_transform(X_test), y_test      # Reverse scale the test set
X1, X2 = np.meshgrid(                                     # Create coordinate grid
    np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.25),
    np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.25)
)
Z = classifier.predict(sc.transform(
    np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)  # Predict for grid points
plt.contourf(X1, X2, Z, alpha = 0.75,
             cmap = ListedColormap(['#FA8072', '#1E90FF']))  # Plot decision boundary
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# Define scatter colors and plot test set points
colors = ['#FA8072', '#1E90FF']
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                color = colors[i], label = j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
