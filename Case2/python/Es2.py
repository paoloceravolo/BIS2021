import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import seaborn as sns

df = pd.DataFrame(np.random.randint(2,size=(100,6)), columns=list(['Analgesics', 'Antibiotic', 'Anticoagulant', 'Antidepressant', 'Anticancer', 'Antiepileptic']))

#print(df.sum(axis=1))

df['Total'] = df.sum(axis=1)

regions = ['Italy','France','Switzerland','Austria','Slovenia']

df['Region'] = random.choices(regions, k=100)

group = df.groupby('Region').agg('sum')

print(group)

sns.heatmap(group, vmin=0, vmax=30, cmap="Reds")
plt.savefig('heatmap.jpg')