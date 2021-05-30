# This file was original a Jupyter Notebook but was copied into a python file

import pandas as pd
import numpy as np
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf



bottle = pd.read_csv('bottle.csv', low_memory=False)
pd.set_option('display.float_format', lambda x: '%f'%x)



bottle['Depthm'] = pd.to_numeric(bottle['Depthm'])
bottle['T_degC'] = pd.to_numeric(bottle['T_degC'])



sub1 = bottle.copy()
sub1 = sub1[(sub1['Depthm'] >= 0) & (sub1['Depthm'] <= 2000)]
sub1 = sub1[(sub1['T_degC'] >= 0) & (sub1['T_degC'] <= 20)]



sub2 = sub1.copy()
sub2['Depthm'] = pd.cut(sub2.Depthm, [0, 500, 1000, 1500, 2000])
sub2['T_degC'] = pd.cut(sub2.T_degC, [0, 5, 10, 15, 20])



sub2.head()



ct = pd.crosstab(sub2['Depthm'], sub2['T_degC'])
print(ct) 



colsum = ct.sum(axis=0)
colpct = ct/colsum
print(colpct)



sub1['Depthm'] = pd.cut(sub1.Depthm, [0, 500, 1000, 1500, 2000])

sns.catplot(x='Depthm', y='T_degC', data=sub1, kind='bar', ci=None)
plt.xlabel('Depth')
plt.ylabel('Temperature')



print('chi-square value, p value, expected counts')
cs1 = scipy.stats.chi2_contingency(ct)
print(cs1)



sub3 = bottle.copy()
cat = [0, 500, 1000, 1500, 2000]

for x in range(0,len(cat)-1):
    for y in range(x+1,len(cat)):
        sub3['temp'] = sub3['Depthm'].map({cat[x]:cat[x], cat[y]:cat[y]})
        cont = pd.crosstab(sub3['T_degC'], sub3['temp'])
        cs = scipy.stats.chi2_contingency(cont)
        print('\n', cat[x], ' versus ', cat[y], 'Chi value: ', cs[0], '\t\tp value: ', cs[1])