import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import pi

df = pd.read_csv('https://raw.githubusercontent.com/paoloceravolo/BIS2021/main/Case3/M1-L3/data.csv', sep=';')

df.timestamp = pd.to_datetime(df.timestamp) 

count_topics = df.groupby('topics').agg({\
'topics': 'count',\
'sentiment': 'sum',\
'irony': 'sum',\
'timestamp': lambda x: x.max() - x.min()\
})
count_topics = count_topics.rename(\
	{'topics': 'occurences', 'timestamp': 'duration'}, axis='columns')

count_topics['duration'] = count_topics['duration'].dt.seconds.astype('int32')

count_topics = count_topics.sort_values('duration', ascending=False)

print(count_topics)

# computing z-scores iterating on the columns 
for col in count_topics.columns: 
    count_topics[col] = (count_topics[col]-count_topics[col].mean())/count_topics[col].std()
print(count_topics.head(6))

df = count_topics.head(4).reset_index()
df = df.rename({'topics': 'group'}, axis='columns')
print(df)

# ------- PART 1: Create background
 
# number of variable
categories=list(df)[1:]
N = len(categories)
 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
 
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
 
# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
 
# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories)
 
# Draw ylabels
ax.set_rlabel_position(0)
# set the dimension of the radar
plt.yticks([-1,0,1,2], ["-1","0","1", "2"], color="grey", size=7)
plt.ylim(-1,3)
 
 
# ------- PART 2: Add plots
 
# Plot each individual = each line of the data
# I don't do a loop, because plotting more than 3 groups makes the chart unreadable
 
# Ind1
values=df.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label=count_topics.index[0])
ax.fill(angles, values, 'b', alpha=0.1)
 
# Ind2
values=df.loc[1].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label=count_topics.index[1])
ax.fill(angles, values, 'r', alpha=0.1)

# Ind3
values=df.loc[2].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label=count_topics.index[2])
ax.fill(angles, values, 'g', alpha=0.1)

# Ind4
values=df.loc[3].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label=count_topics.index[3])
ax.fill(angles, values, 'm', alpha=0.1)
 
# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

# Visualize
plt.show()