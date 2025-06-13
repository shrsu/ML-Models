import numpy as np
# Imports NumPy for numerical operations
import matplotlib.pyplot as plt
# Imports Matplotlib for data visualization (not used in this script but commonly included)
import pandas as pd
# Imports pandas for data manipulation and analysis

# Loads the dataset into a pandas DataFrame without any header row
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

# Initializes an empty list to store transactions
transactions = []
# Loops through all 7501 rows (transactions)
for i in range(0, 7501):
    # For each transaction, appends a list of items (converted to strings) from 20 columns
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# Imports the apriori function from the apyori package
from apyori import apriori

# Applies the apriori function to find association rules with the following constraints:
# min_support: itemsets must appear in at least 0.3% of the transactions
# min_confidence: rules must have at least 20% confidence
# min_lift: rules must have at least 3x improvement over random co-occurrence
# min_length and max_length: only consider itemsets of size 2
rules = apriori(transactions=transactions, 
                min_support=0.003, 
                min_confidence=0.2, 
                min_lift=3, 
                min_length=2, 
                max_length=2)

# Converts the rules generator into a list for easier handling and display
results = list(rules)

# Defines a helper function to extract relevant parts of each rule
def inspect(results):
    # Extracts the left-hand side (LHS) item of the rule
    lhs = [tuple(result[2][0][0])[0] for result in results]
    # Extracts the right-hand side (RHS) item of the rule
    rhs = [tuple(result[2][0][1])[0] for result in results]
    # Extracts the support for each rule
    supports = [result[1] for result in results]
    # Returns a zipped list of (LHS, RHS, Support)
    return list(zip(lhs, rhs, supports))

# Creates a DataFrame from the inspected results with proper column names
resultsinDataFrame = pd.DataFrame(inspect(results), columns=['Product 1', 'Product 2', 'Support'])

# Displays the top 10 rules with the highest support values
resultsinDataFrame.nlargest(n=10, columns='Support')
