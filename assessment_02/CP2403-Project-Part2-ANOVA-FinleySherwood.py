# This file was original a Jupyter Notebook but was copied into a python file

import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi 
import matplotlib.pyplot as plt



pd.set_option('display.float_format', lambda x: '%.2f'%x)
bottle = pd.read_csv('bottle.csv', low_memory=False)
bottle.head()



bottle['Depthm'] = pd.to_numeric(bottle['Depthm'], errors='coerce')
bottle['O2ml_L'] = pd.to_numeric(bottle['O2ml_L'], errors='coerce')



sub1 = bottle.copy()
sub1 = sub1[(sub1['Depthm'] >= 0) & (sub1['Depthm'] <= 3500)]
sub1 = sub1[(sub1['O2ml_L'] >= 0) & (sub1['O2ml_L'] <= 4)]



sub1['Depthm'] = pd.cut(sub1.Depthm, [500, 1000, 1500, 2000, 2500, 3000, 3500])
sub1['Depthm'] = sub1['Depthm'].astype('category')



sns.boxplot(x='Depthm', y='O2ml_L', data=sub1)
plt.xlabel('Depth')
plt.ylabel('Dissolved Oxygen Concentration')



model1 = smf.ols(formula='O2ml_L ~ C(Depthm)', data=sub1).fit()
print(model1.summary())



print('means for dissolved oxygen concentration grouped by depth')
means = sub1.groupby('Depthm').mean()
print(means['O2ml_L'])



print('Standard deviation for dissolved oxygen concentration grouped by depth')
means = sub1.groupby('Depthm').std()
print(means['O2ml_L'])