# This file was original a Jupyter Notebook but was copied into a python file

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import scipy
import statsmodels.api as sm



pd.set_option('display.float_format', lambda x: '%.2f'%x)
bottle = pd.read_csv('bottle.csv', low_memory=False)
bottle.head()



bottle['T_degC'] = pd.to_numeric(bottle['T_degC'], errors='coerce')
bottle['PO4uM'] = pd.to_numeric(bottle['PO4uM'], errors='coerce')
bottle['R_SALINITY'] = pd.to_numeric(bottle['R_SALINITY'], errors='coerce')



sub1 = bottle.copy()
sub1 = sub1[(sub1['T_degC'] >= 5) & (sub1['T_degC'] <= 20)]
sub1 = sub1[(sub1['PO4uM'] >= 1) & (sub1['PO4uM'] <= 2)]
sub1 = sub1[(sub1['R_SALINITY'] >= 33) & (sub1['R_SALINITY'] <= 34)]



sns.displot(bottle['T_degC'].dropna(), kde=False)



sns.displot(sub1['PO4uM'].dropna(), kde=False) # Using displot instead of distplot because it is deprecated



sns.displot(sub1['R_SALINITY'].dropna(), kde=False) # Using displot instead of distplot because it is deprecated



sub1['T_degC'].describe()



plt.figure()

scat1 = sns.regplot(x='T_degC', y='PO4uM', fit_reg=True, order=3, data=sub1)
plt.xlabel('Temperature')
plt.ylabel('Phosphate Concentration')
plt.title('Scatterplot for the Association Between Temperature and Phosphate Concentraion')



# There is a strong negative correlation between the two variables
print('Association between Phosphate and Temperature')
print(scipy.stats.pearsonr(sub1['T_degC'], sub1['PO4uM']))



plt.figure()

scat1 = sns.regplot(x='R_SALINITY', y='PO4uM', fit_reg=True, order=2, data=sub1)
plt.xlabel('Salt Concentration')
plt.ylabel('Phosphate Concentration')
plt.title('Scatterplot for the Association Between Temperature and Salt Concentraion')



# There is a strong positive correlation between the two variables
print('Association between Salt and Phosphate')
print(scipy.stats.pearsonr(sub1['R_SALINITY'], sub1['PO4uM']))



plt.figure()

scat1 = sns.regplot(x='T_degC', y='PO4uM', data=sub1, fit_reg=True)
plt.xlabel('Temperature')
plt.ylabel('Phosphate Concentration')



sub1['T_degC_c'] = sub1['T_degC'] - sub1['T_degC'].mean()
sub1['PO4uM_c'] = sub1['PO4uM'] - sub1['PO4uM'].mean()
sub1['R_SALINITY_c'] = sub1['R_SALINITY'] - sub1['R_SALINITY'].mean()
sub1.head()



reg = smf.ols('PO4uM_c ~ T_degC_c + R_SALINITY_c', data=sub1).fit()
print(reg.summary())



sm.qqplot(reg.resid, line='r')



stdres = pd.DataFrame(reg.resid_pearson)

plt.figure()

plt.plot(stdres, 'o', ls='None')
l = plt.axhline(y=0, color='r')
plt.ylabel('Standardized Residual')
plt.xlabel('Observation Number')



percentage_over2sd = (np.count_nonzero(stdres[0] > 2) + np.count_nonzero(stdres[0] < -2))/len(stdres)*100
print(percentage_over2sd)



percentage_over2_5sd = (np.count_nonzero(stdres[0] > 2.5) + np.count_nonzero(stdres[0] < -2.5))/len(stdres)*100
print(percentage_over2_5sd)