import matplotlib.pyplot as plt
import pandas as pd

regions = ['Italy','France','Switzerland','Austria','Slovenia']
sold = [30,24,23,19,12]
pv = [150,140,130,150,155]

data = {'Regions': regions, 'Sold': sold}

df = pd.DataFrame(data, index=regions)

print(df)

plt.subplot(2, 1, 1)
plt.bar(df['Regions'], df['Sold'], color='green')
plt.xlabel('Regions')
plt.ylabel('Products Sold in 2020')
plt.title('Sold per region')

gen_value = [v*pv[i] for i,v in enumerate(df['Sold'])]
df['Value'] = gen_value

#plt.show()

plt.subplot(2, 1, 2)
plt.bar(df['Regions'], df['Value'], color='red', label='Generated value')
plt.xlabel('Regions')
plt.ylabel('Value generated in 2020')
plt.title('Value generated per region')

plt.show()



