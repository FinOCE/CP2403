import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm



pd.set_option('display.float_format', lambda x:'%.2f'%x)

gapminder = pd.read_csv('gapminder.csv', low_memory=False)
gapminder.head()



gapminder['oilperperson'] = pd.to_numeric(gapminder['oilperperson'],errors='coerce')
gapminder['relectricperperson'] = pd.to_numeric(gapminder['relectricperperson'],errors='coerce')
gapminder['co2emissions'] = pd.to_numeric(gapminder['co2emissions'],errors='coerce')



# Scenario 1: Linear and multiple
sub1 = gapminder[['oilperperson', 'relectricperperson', 'co2emissions']].dropna()
sub1.head()



sub1['oilperperson_c'] = (sub1['oilperperson'] - sub1['oilperperson'].mean())
sub1['relectricperperson_c'] = (sub1['relectricperperson'] - sub1['relectricperperson'].mean())
sub1['co2emissions_c'] = (sub1['co2emissions'] - sub1['co2emissions'].mean())

sub1.head()



reg1 = smf.ols('oilperperson_c ~ relectricperperson_c + oilperperson_c', data=sub1).fit()
print(reg1.summary())



# Scenario 2: Linear
gapminder['employrate'] = pd.to_numeric(gapminder['employrate'], errors='coerce')
sub2 = gapminder[['relectricperperson', 'employrate']].dropna()
sub2.head()



#%matplotlib inline # Causing error when copy to python file from jupyter notebook

plt.figure()
scat1 = sns.regplot(x='employrate', y='relectricperperson', fit_reg=True, data=sub2)

plt.xlabel('Electricity Use Per Person')
plt.ylabel('Employment Rate')



sub2['relectricperperson_c'] = (sub2['relectricperperson'] - sub2['relectricperperson'].mean())
sub2['employrate_c'] = (sub2['employrate'] - sub2['employrate'].mean())

sub2.head()



reg2 = smf.ols('employrate_c ~ relectricperperson_c', data=sub2).fit()
print(reg2.summary())



# Scenario 3: Polynomial
plt.figure()
scat1 = sns.regplot(x='employrate', y='relectricperperson', order=2, data=sub2)

plt.xlabel('Electricity Use Per Person')
plt.ylabel('Employment Rate')



reg2 = smf.ols('employrate_c ~  I(relectricperperson_c**2)', data=sub2).fit()
print(reg2.summary())



# Scenario 4: Multiple and polynomial
sub3 = gapminder[['oilperperson', 'relectricperperson', 'co2emissions','employrate']].dropna()
sub3.head()



sub3['employrate_c'] = (sub3['employrate'] - sub3['employrate'].mean())
sub3['oilperperson_c'] = (sub3['oilperperson'] - sub3['oilperperson'].mean())
sub3['relectricperperson_c'] = (sub3['relectricperperson'] - sub3['relectricperperson'].mean())
sub3['co2emissions_c'] = (sub3['co2emissions'] - sub3['co2emissions'].mean())



reg3 = smf.ols('employrate_c ~ oilperperson_c + co2emissions_c + I(relectricperperson_c**2)', data=sub3).fit()
print(reg3.summary())



fig4 = sm.qqplot(reg3.resid, line='r')



stdres = pd.DataFrame(reg3.resid_pearson)

plt.figure()
plt.plot(stdres, 'o', ls='None')
l = plt.axhline(y=0, color='r')
plt.ylabel('Standardized Residual')
plt.xlabel('Observation Number')



percentage_over2sd = (np.count_nonzero(stdres[0] > 2) + np.count_nonzero(stdres[0] < -2))/len(stdres)*100
print(percentage_over2sd)



percentage_over2_5sd = (np.count_nonzero(stdres[0] > 2.5) + np.count_nonzero(stdres[0] < -2.5))/len(stdres)*100
print(percentage_over2_5sd)



# Scenario 5
reg4 = smf.ols('employrate_c ~ I(oilperperson_c**2) + co2emissions_c + relectricperperson_c', data=sub3).fit()
print(reg4.summary())



fig5 = sm.qqplot(reg4.resid, line='r')



stdres = pd.DataFrame(reg4.resid_pearson)

plt.figure()
plt.plot(stdres, 'o', ls='None')
l = plt.axhline(y=0, color='r')
plt.ylabel('Standardized Residual')
plt.xlabel('Observation Number')



percentage_over2sd = (np.count_nonzero(stdres[0] > 2) + np.count_nonzero(stdres[0] < -2))/len(stdres)*100
print(percentage_over2sd)



percentage_over2_5sd = (np.count_nonzero(stdres[0] > 2.5) + np.count_nonzero(stdres[0] < -2.5))/len(stdres)*100
print(percentage_over2_5sd)