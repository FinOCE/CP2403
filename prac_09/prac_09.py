import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf 



nesarc = pd.read_csv('nesarc - large.csv', low_memory=False)
pd.set_option('display.float_format', lambda x: '%f'%x)



nesarc['S2AQ5B'] = pd.to_numeric(nesarc['S2AQ5B'], errors='coerce')
nesarc['S2AQ5D'] = pd.to_numeric(nesarc['S2AQ5D'], errors='coerce')
nesarc['S2AQ5A'] = pd.to_numeric(nesarc['S2AQ5A'], errors='coerce')
nesarc['S2BQ1B1'] = pd.to_numeric(nesarc['S2BQ1B1'], errors='coerce')
nesarc['AGE'] = pd.to_numeric(nesarc['AGE'], errors='coerce')



sub1 = nesarc[(nesarc['AGE'] >= 26) & (nesarc['AGE'] <= 50) & (nesarc['S2AQ5A'] == 1)]
sub2 = sub1.copy()



sub2['S2AQ5D'] = sub2['S2AQ5D'].replace(99, np.nan)

sub2['S2AQ5B'] = sub2['S2AQ5B'].replace(8, np.nan)
sub2['S2AQ5B'] = sub2['S2AQ5B'].replace(9, np.nan)
sub2['S2AQ5B'] = sub2['S2AQ5B'].replace(10, np.nan)
sub2['S2AQ5B'] = sub2['S2AQ5B'].replace(99, np.nan)

sub2['S2BQ1B1']=sub2['S2BQ1B1'].replace(9, np.nan)



recode2 = {1:30, 2:26, 3:14, 4:8, 5:4, 6:2.5, 7:1}
sub2['BEER_FEQMO'] = sub2['S2AQ5B'].map(recode2)

recode3 = {2:0, 1:1}
sub2['S2BQ1B1'] = sub2['S2BQ1B1'].map(recode3)



sub2['NUMBEERMO_EST'] = sub2['BEER_FEQMO'] * sub2['S2AQ5D']



# Scenario 1
reg1 = smf.ols('NUMBEERMO_EST ~ S2BQ1B1', data=sub2).fit()
print(reg1.summary())



sub3 = sub2[['NUMBEERMO_EST', 'S2BQ1B1']].dropna()

print('Mean')
ds1 = sub3.groupby('S2BQ1B1').mean()
print(ds1)

print('Standard deviation')
ds2 = sub3.groupby('S2BQ1B1').std()
print(ds2)



#%matplotlib inline

sns.factorplot(x='S2BQ1B1', y='NUMBEERMO_EST', data=sub3, kind='bar', ci=None)

plt.xlabel('Beer Dependence')
plt.ylabel('Mean Number of Beers consumed')



# Scenario 2
lreg1 = smf.logit(formula='S2BQ1B1 ~ GENAXLIFE', data=sub2).fit()
print(lreg1.summary())



params = lreg1.params
conf = lreg1.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print(np.exp(conf))



# Scenario 3
sub2['DYSLIFE'] = pd.to_numeric(sub2['DYSLIFE'], errors='coerce')



lreg2 = smf.logit(formula='S2BQ1B1 ~ GENAXLIFE + DYSLIFE', data=sub2).fit()
print(lreg2.summary())



params = lreg2.params
conf = lreg2.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print(np.exp(conf))



# Scenario 4
def PANIC(x1):
    if (
        (x1['S6Q1'] == 1 and x1['S6Q2'] == 1) or (x1['S6Q2'] == 1 and x1['S6Q3'] == 1) or 
        (x1['S6Q3'] == 1 and x1['S6Q61'] == 1) or (x1['S6Q61'] == 1 and x1['S6Q62'] == 1) or 
        (x1['S6Q62'] == 1 and x1['S6Q63'] == 1) or (x1['S6Q63'] == 1 and x1['S6Q64'] == 1) or 
        (x1['S6Q64'] == 1 and x1['S6Q65'] == 1) or (x1['S6Q65'] == 1 and x1['S6Q66'] == 1) or 
        (x1['S6Q66'] == 1 and x1['S6Q67'] == 1) or (x1['S6Q67'] == 1 and x1['S6Q68'] == 1) or 
        (x1['S6Q68'] == 1 and x1['S6Q69'] == 1) or (x1['S6Q69'] == 1 and x1['S6Q610'] == 1) or 
        (x1['S6Q610'] == 1 and x1['S6Q611'] == 1) or (x1['S6Q611'] == 1 and x1['S6Q612'] == 1) or 
        (x1['S6Q612'] == 1 and x1['S6Q613'] == 1) or (x1['S6Q613'] == 1 and x1['S6Q7'] == 1) or 
        (x1['S6Q7'] == 1)
    ):
        return 1
    else:
        return 0
sub2['PANIC'] = sub1.apply(lambda x1: PANIC(x1), axis=1)
c7 = sub2['PANIC'].value_counts(sort=False, dropna=False)
print(c7)



lreg3 = smf.logit(formula='S2BQ1B1 ~ PANIC', data=sub2).fit()
print(lreg3.summary())



print('Odds Ratios')
params = lreg3.params
conf = lreg3.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print(np.exp(conf))



# Scenario 5
lreg4 = smf.logit(formula='S2BQ1B1 ~ PANIC + DYSLIFE', data=sub2).fit()
print(lreg4.summary())



print('Odds Ratios')
params = lreg4.params
conf = lreg4.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print(np.exp(conf))