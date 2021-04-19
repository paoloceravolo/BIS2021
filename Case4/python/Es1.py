import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
from mlxtend.frequent_patterns import association_rules


df = pd.read_csv('./saleslog.csv', parse_dates=['Date'])

print(df)

itemlist = df.groupby('Member_number')['itemDescription'].apply(list)\
.reset_index().rename(columns={'itemDescription':'Selected Items'})

print(itemlist)

matrix = [itemlist['Selected Items'][i] for i,v in enumerate(itemlist['Selected Items'])]

print(matrix)

te = TransactionEncoder()
te_ary = te.fit(matrix).transform(matrix)
df2 = pd.DataFrame(te_ary, columns=te.columns_)

print(df2)

frequent_itemsets = fpgrowth(df2, min_support=0.1, use_colnames=True)

print(frequent_itemsets)

rules = association_rules(frequent_itemsets, metric='lift', min_threshold=0.1)

print(rules)











