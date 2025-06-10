# Import necessary libraries
import numpy as np                     # For numerical operations (not used directly in this code)
import matplotlib.pyplot as plt        # For plotting (not used here but commonly imported)
import pandas as pd                    # For handling data in DataFrames

# Load the dataset (each row is a transaction, each column is an item)
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

# Convert the dataset into a list of transactions (list of lists)
transactions = []
for i in range(0, 7501):  # Loop through each row (transaction)
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])  # Convert items to strings

# Import the apriori algorithm from the apyori package
from apyori import apriori

# Apply the Apriori algorithm to extract association rules
rules = apriori(
    transactions=transactions,  # The list of transactions
    min_support=0.003,          # Itemset should appear in at least 0.3% of transactions
    min_confidence=0.2,         # Rule must be correct at least 20% of the time
    min_lift=3,                 # Lift must be at least 3 (positive correlation)
    min_length=2,               # Rule must have at least 2 items
    max_length=2                # Rule must have at most 2 items
)

# Convert the rules generator object into a list for easy access
results = list(rules)

# Define a function to parse the results into readable format
def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]           # Extract Left Hand Side (antecedent)
    rhs = [tuple(result[2][0][1])[0] for result in results]           # Extract Right Hand Side (consequent)
    supports = [result[1] for result in results]                      # Extract support values
    confidences = [result[2][0][2] for result in results]             # Extract confidence values
    lifts = [result[2][0][3] for result in results]                   # Extract lift values
    return list(zip(lhs, rhs, supports, confidences, lifts))         # Combine into tuples

# Create a DataFrame to display the extracted rules in tabular format
resultsinDataFrame = pd.DataFrame(
    inspect(results),
    columns=['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift']
)

# Display the DataFrame containing all the rules (unsorted)
resultsinDataFrame

# Display the top 10 rules sorted by highest lift (strongest association)
resultsinDataFrame.nlargest(n=10, columns='Lift')
