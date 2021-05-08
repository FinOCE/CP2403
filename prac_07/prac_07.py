import pandas as pd
import numpy as np
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf



pd.set_option('display.float_format', lambda x:'%.2f'%x)

gapminder = pd.read_csv('gapminder.csv', low_memory=False)
gapminder.head()



gapminder['oilperperson'] = pd.to_numeric(gapminder['oilperperson'],errors='coerce')
gapminder['relectricperperson'] = pd.to_numeric(gapminder['relectricperperson'],errors='coerce')
gapminder['co2emissions'] = pd.to_numeric(gapminder['co2emissions'],errors='coerce')



gapminder_clean=gapminder.dropna()



# Scatter plot to show association between relectricperperson (x) and oilperperson (y)
%matplotlib inline
plt.figure()
scat1 = sns.regplot(x='relectricperperson', y='oilperperson', fit_reg=False, data=gapminder_clean)

plt.xlabel('Electrcity Use Per Person')
plt.ylabel('Oil Use Per Person')
plt.title('Scatterplot for the Association Between Electrcity Use Per Person' + '\n' + 'and Oil Use Per Person')



# Pearson correlation - relectricperperson (x) and oilperperson (y)
print('association between relectricperperson and oilperperson')
print(scipy.stats.pearsonr(gapminder_clean['relectricperperson'], gapminder_clean['oilperperson']))



# Scatter plot to show association between co2emissions (x) and oilperperson (y)
%matplotlib inline
plt.figure()
scat2 = sns.regplot(x='co2emissions', y='oilperperson', fit_reg=False, data=gapminder_clean)

plt.xlabel('CO2 Emissions')
plt.ylabel('Oil Use Per Person')
plt.title('Scatterplot for the Association' + '\n' + 'CO2 Emission and Oil Use Per Person')



# Pearson correlation - co2emissions (x) and oilperperson (y)
print('association between co2emissions and oilperperson')
print(scipy.stats.pearsonr(gapminder_clean['co2emissions'], gapminder_clean['oilperperson']))



# Scatter plot with regression to show relationship between relectricperperson (x) and oilperperson (y) - with regression line
%matplotlib inline
scat1 = sns.regplot(x='relectricperperson', y='oilperperson', fit_reg=True, data=gapminder_clean)

plt.xlabel('Electricity Use Per Person')
plt.ylabel('Oil Use Per Person')
plt.title('Scatterplot for the Association Between Electricity Use Per Person' + '\n' + 'and Oil Use Per Person')



# Regression analysis to show association between relectricperperson (x) and oilperperson (y)
print('OLS regression model for the association between Electric Use Per Person and Oil Per Person')
reg1 = smf.ols('oilperperson ~ relectricperperson', data=gapminder_clean).fit()
print(reg1.summary())



# Residual plot - regression analysis between relectricperperson (x) and oilperperson (y)
%matplotlib inline
scat1 = sns.residplot(x='relectricperperson', y='oilperperson',  data=gapminder_clean)

plt.xlabel('Electricity Use Per Person')
plt.ylabel('Oil Use Per Person')
plt.title('Residual Plot')



# Scatter plot with regression to show association between co2emissions (x) and oilperperson (y) - with regression line
plt.figure()
scat2 = sns.regplot(x='co2emissions', y='oilperperson', fit_reg=True, data=gapminder)

plt.xlabel('CO2 Emission')
plt.ylabel('Oil Use Per Person')
plt.title('Scatterplot for the Association' + '\n' + 'Between CO2 Emission and Oil Use Per Person')



# Regression analysis to show association between co2emissions (x) and oilperperson (y)
print('OLS regression model for the association between CO2 emission and Oil Use Per Person')
reg1 = smf.ols('oilperperson ~ co2emissions', data=gapminder_clean).fit()
print(reg1.summary())



# Residual plot - regression analysis between co2emissions (x) and oilperperson (y)
%matplotlib inline
scat1 = sns.residplot(x='co2emissions', y='oilperperson',  data=gapminder_clean)

plt.xlabel('CO2 Emission')
plt.ylabel('Oil Per Person')
plt.title('Residual Plot')



# Regression with 3 variables
def co2emissionsgrp(row):
    if row['co2emissions'] <= 1846084167:
        return 1
    elif row['co2emissions'] <= 7993752800:
        return 2
    elif row['co2emissions'] > 7993752800:
        return 3



# When attempting to follow the set structure with:

# gapminder_clean['co2emissionsgrp'] = gapminder_clean.apply(lambda row: co2emissionsgrp(row), axis=1)

# I ran into the following error:

# <ipython-input-25-f726a11f984d>:1: SettingWithCopyWarning: 
# A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead

# I could not figure out a solution, but since it mentioned the issue involving the copy, I decided to
# define a new variable to be used as the DataFrame for the rest of the questions, since they all use
# sub from the 'co2emissionsgrp' property in 'gapminder_clean' DataFrame.

co2emissionsgrp = gapminder_clean.apply(lambda row: co2emissionsgrp(row), axis=1)

# Print the number of countries in each group of CO2 emission
chk1 = co2emissionsgrp.value_counts(sort=False, dropna=False)
print(chk1)



# Divide gapminder_clean into 3 dataframes, each dataframe representing rows of data in low, medium and high CO2 Emission
sub1 = gapminder_clean[(co2emissionsgrp == 1)]
sub2 = gapminder_clean[(co2emissionsgrp == 2)]
sub3 = gapminder_clean[(co2emissionsgrp == 3)]



# Scatter plot with regression analysis to show association between electricity use per person (x) and oilperperson (y) for low CO2 emission countries
%matplotlib inline
scat1 = sns.regplot(x='relectricperperson', y='oilperperson', data=sub1)

plt.xlabel('Electricity Use Per Person')
plt.ylabel('Oil Use Per Person')
plt.title('Scatterplot for the Association Between Electricity Use Per Person and' +  '\n' + 'Oil Use Per Person for LOW CO2 emissions countries')
print(scat1)



# Regression analysis to show association between electricity use per person (x) and oilperperson (y) for low CO2 emission countries
print('OLS regression model for the association between Electricty Use Per Person and Oil Use Per Person for' + '\n' + 'LOW CO2 Emission countries')
reg1 = smf.ols('oilperperson ~ relectricperperson', data=sub1).fit()
print(reg1.summary())



# Residual plot - regression analysis between relectricperperson (x) and oilperperson (y) for Low CO2 emission countries
%matplotlib inline
scat1 = sns.residplot(x='relectricperperson', y='oilperperson', data=sub1)

plt.xlabel('Electricity Use Per Person')
plt.ylabel('Oil Use Per Person')
plt.title('Residual Plot - LOW CO2 Emission Countries')



# Scatter plot with regression analysis to show association between electricity use per person (x) and oilperperson (y) for medium CO2 emission countries
%matplotlib inline
scat1 = sns.regplot(x='relectricperperson', y='oilperperson', data=sub2)

plt.xlabel('Electricity Use Per Person')
plt.ylabel('Oil Use Per Person')
plt.title('Scatterplot for the Association Between Electricity Use Per Person and' +  '\n' + 'Oil Use Per Person for MEDIUM CO2 emissions countries')
print(scat1)



print('OLS regression model for the association between Electricty Use Per Person and Oil Use Per Person for' + '\n' + 'MEDIUM CO2 Emission countries')
reg1 = smf.ols('oilperperson ~ relectricperperson', data=sub2).fit()
print(reg1.summary())



# Residual plot - regression analysis between relectricperperson (x) and oilperperson (y) for Medium CO2 emission countries
%matplotlib inline
scat1 = sns.residplot(x='relectricperperson', y='oilperperson', data=sub2)

plt.xlabel('Electricity Use Per Person')
plt.ylabel('Oil Use Per Person')
plt.title('Residual Plot - Medium CO2 Emission Countries')



# Scatter plot with regression analysis to show association between electricity use per person (x) and oilperperson (y) for high CO2 emission countries
%matplotlib inline
scat1 = sns.regplot(x='relectricperperson', y='oilperperson', data=sub3)

plt.xlabel('Electricity Use Per Person')
plt.ylabel('Oil Use Per Person')
plt.title('Scatterplot for the Association Between Electricity Use Per Person and' +  '\n' + 'Oil Use Per Person for HIGH CO2 emissions countries')
print(scat1)



print('OLS regression model for the association between Electricty Use Per Person and Oil Use Per Person for' + '\n' + 'HIGH CO2 Emission countries')
reg1 = smf.ols('oilperperson ~ relectricperperson', data=sub3).fit()
print(reg1.summary())



# Residual plot - regression analysis between relectricperperson (x) and oilperperson (y) for High CO2 emission countries
%matplotlib inline
scat1 = sns.residplot(x='relectricperperson', y='oilperperson', data=sub3)

plt.xlabel('Electricity Use Per Person')
plt.ylabel('Oil Use Per Person')
plt.title('Residual Plot - High CO2 Emission Countries')