## 
## Abstracting the item description
## (i) import data (ii) merge datsets creating a new feature (iii) list items by user (iv) compute frequent sequences (v) compute association rules on the new feature instead of items
##

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
from mlxtend.frequent_patterns import association_rules

#PrefixSpan: Mining Sequential Patterns by Prefix-Projected Growth.
#Jian Pei, Jiawei Han, Behzad Mortazavi-Asl, Helen Pinto, Qiming Chen, Umeshwar Dayal, Meichun Hsu.
#Proceedings of the 17th International Conference on Data Engineering, 2001.
#https://github.com/chuanconggao/PrefixSpan-py
from prefixspan import PrefixSpan

## Import as Dataframe
# Adding the parse_dates=['date'] argument will make the date column to be parsed as a date field.
df1 = pd.read_csv('./saleslog.csv', parse_dates=['Date'])
df2 = pd.read_csv('./wine_description.csv')

df_merge = pd.merge(df1, df2, left_on='itemDescription', right_on='name')

print(df_merge)

# create a new column based on other columns
df_merge['Origin-Type'] = df_merge.apply(lambda row: row.origin +'-'+ row.type, axis=1)


#countMembers = df_merge.groupby('Member_number').count()
#print(countMembers.sort_values(by='itemDescription'))

## Convert values of each group into a list
itemlist = df_merge.groupby('Member_number')['Origin-Type'].apply(list)\
.reset_index().rename(columns={'Origin-Type':'Selected Items'})

print(itemlist)

# a matrix is requires as imput to the AR functions
# because print(itemlist['Selected Items'].values) generate a set of list()
# we have to create a matrix by listing the entire colum
#matrix = np.array([itemlist['Selected Items'][i] for i,v in enumerate(itemlist['Selected Items'])])

matrix = [itemlist['Selected Items'][i] for i,v in enumerate(itemlist['Selected Items'])]

#print(matrix)


## Compute association rules
#te = TransactionEncoder()

#te_ary = te.fit(matrix).transform(matrix)
#df_trans = pd.DataFrame(te_ary, columns=te.columns_)

#print(df2)

#frequent_itemsets = fpgrowth(df_trans, min_support=0.2, use_colnames=True)

#print(frequent_itemsets)

#rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.1)
#print(rules)

## Compute Frequent Patterns

ps = PrefixSpan(matrix)

print(ps.topk(5, generator=True))