import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load the data
data = pd.read_csv('france.csv')

# Convert the data into the appropriate format for association rule mining
transaction_matrix = (data
                      .groupby(['InvoiceNo', 'Description'])['Quantity']
                      .sum().unstack().reset_index().fillna(0)
                      .set_index('InvoiceNo'))

# Convert any positive values to 1
transaction_matrix = transaction_matrix.applymap(lambda x: 1 if x > 0 else 0)

# Apply the Apriori algorithm
frequent_itemsets = apriori(transaction_matrix, min_support=0.01,  use_colnames=True)

# Generate the association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Print the rules
print("Association Rules:")
print(rules)

# Print the confidence report
print("\nConfidence Report:")
print(rules[['antecedents', 'consequents', 'confidence']])
