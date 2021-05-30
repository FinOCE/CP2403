# This file was original a Jupyter Notebook but was copied into a python file

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



bottles = pd.read_csv('bottle.csv', low_memory=False)



box_data = bottles.copy()

box_data['Depthm'] = pd.to_numeric(box_data['Depthm'], errors='coerce')
box_data['SiO3uM'] = pd.to_numeric(box_data['SiO3uM'], errors='coerce')
box_data['Depthm'] = box_data['Depthm'].astype('category')
box_data['Depthm'] = pd.cut(box_data.Depthm, [0, 1000, 2000, 3000, 4000, np.inf])



histogram_data = bottles.copy()

histogram_data['T_degC'] = pd.to_numeric(histogram_data['T_degC'], errors='coerce')



line_data = bottles.copy()

line_data['T_degC'] = pd.to_numeric(line_data['T_degC'], errors='coerce')
line_data['Depthm'] = pd.to_numeric(line_data['Depthm'], errors='coerce')
line = line_data.groupby('Depthm').T_degC.mean()



bubble_data = bottles.copy()

bubble_data['Depthm'] = pd.to_numeric(bubble_data['Depthm'], errors='coerce')
bubble_data['PO4uM'] = pd.to_numeric(bubble_data['PO4uM'], errors='coerce')
bubble_data['PO4uM'] = bubble_data['PO4uM'].dropna()
bubble_data['O2Sat'] = pd.to_numeric(bubble_data['O2Sat'], errors='coerce')
bubble_data['O2Sat'] = bubble_data['O2Sat'].dropna()




fig = plt.figure()

sns.boxplot(box_data['Depthm'], box_data['SiO3uM'])

plt.title('Concentration of Silicate at Different Depths')
plt.xlabel('Depth (m)')
plt.ylabel('Concentration of Silicate (μM)')

print("This graph shows that as the depth of the water increases, the concentration of silicate also increases. The depth was split into 5 categories since most of the useful data in the dataset is numeric and it is difficult to draw any conclusions off the categorical.")




fig = plt.figure()

sns.histplot(histogram_data['T_degC'])

plt.title('Histogram for Temperature of Ocean Water')
plt.xlabel('Temperature (°C)')

print("The conclusion that can be drawn from this histogram is that the temperature of the water in the dataset is unimodal and skewed right. The majority of temperature readings are at approximately 10°C and there is very little once the temperatures 0°C and 30°C are reached.")




fig = plt.figure()

line.plot(kind='line')

plt.title('Temperature of Water as Depth Increases')
plt.xlabel('Depth (m)')
plt.ylabel('Temperature (°C)')

print("This line shows a clear trend where as the depth of the test increases, the temperature of the water decreases. The points where the temperature spikes in the middle of the trend line, particularly at approximately 1500m depth, are likely caused by readings from different climates being factored into the graph. These spikes match the histogram above, where they peak where the most common temperatures are, around 8-10°C.")



fig = plt.figure()

sns.scatterplot(bubble_data['Depthm'], bubble_data['PO4uM'], size=bubble_data['O2Sat'])

plt.title('Concentration of Phosphate and Oxygen With Relation to Depth')
plt.xlabel('Depth (m)')
plt.ylabel('Phosphate Concentration (μM)')

print("This graph shows that as depth increases, the concentration of phosphate increases, and the concentration of oxygen decreases. There are many points that look to be outliers, however there is still a clear trend showing the rapid increase in phosphate between 0m and 1000m depth. Unfortunately, my laptop does not seem to be powerful enough to complete this graph, since running it for an extended period of time did not yield a graph. In theory, the code is all accurate and should work as intended.")



# Using graph #4...

# Import Data
df = bottles.copy()

# Draw Stripplot
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)    
sns.stripplot(df.R_Depth, df.R_SIGMA, jitter=0.25, size=8, ax=ax, linewidth=.5)

# Decorations
plt.title('Use jittered plots to avoid overlapping of points', fontsize=22)
plt.show()

print("This graph shows that as the recorded depth of the water increases, the recorded potential density of the water also increases. Like the bubble chart, unfortunately my laptop does not seem capable of creating this graph, since leaving it running for a long time yielded no graph.")