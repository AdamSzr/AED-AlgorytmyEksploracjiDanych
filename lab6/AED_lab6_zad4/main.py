import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpmax

transactions = [
    ["Bread", "Milk"],
    ["Bread", "Diapers", "Beer", "Eggs"],
    ["Milk", "Diapers", "Beer", "Cola"],
    ["Bread", "Milk", "Diapers", "Beer"],
    ["Bread", "Milk", "Diapers", "Cola"],
    ["Bread", "Beer", "Cola"],
    ["Bread", "Milk", "Eggs"],
    ["Bread", "Milk"],
    ["Bread", "Eggs"],
    ["Diapers", "Beer", "Cola"],
    ["Milk", "Diapers", "Beer", "Cola"],
    ["Bread", "Milk", "Diapers", "Beer", "Eggs", "Cola"]
]

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

print(fpmax(df, min_support=0.3, use_colnames=True).size)
