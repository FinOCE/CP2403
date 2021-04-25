import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi 
import matplotlib.pyplot as plt



nesarc = pd.read_csv('nesarc.csv', low_memory=False)
pd.set_option('display.float_format', lambda x:'%f'%x)



nesarc['S2AQ5B'] = pd.to_numeric(nesarc['S2AQ5B'], errors='coerce') #convert variable to numeric
nesarc['S2AQ5D'] = pd.to_numeric(nesarc['S2AQ5D'], errors='coerce') #convert variable to numeric
nesarc['S2AQ5A'] = pd.to_numeric(nesarc['S2AQ5A'], errors='coerce') #convert variable to numeric



sub1 = nesarc[(nesarc['AGE'] >= 26) & (nesarc['AGE'] <= 50) & (nesarc['S2AQ5A'] == 1)]
sub2 = sub1.copy()



#SETTING MISSING DATA
sub2['S2AQ5D'] = sub2['S2AQ5D'].replace(99, np.nan)

sub2['S2AQ5B'] = sub2['S2AQ5B'].replace(8, np.nan)
sub2['S2AQ5B'] = sub2['S2AQ5B'].replace(9, np.nan)
sub2['S2AQ5B'] = sub2['S2AQ5B'].replace(10, np.nan)
sub2['S2AQ5B'] = sub2['S2AQ5B'].replace(99, np.nan)



recode2 = {1:30, 2:26, 3:14, 4:8, 5:4, 6:2.5, 7:1}
sub2['BEER_FEQMO'] = sub2['S2AQ5B'].map(recode2)
sub2['BEER_FEQMO'] = pd.to_numeric(sub2['BEER_FEQMO'])



# Creating a secondary variable multiplying the days consumed beer/month and the number of beer/per day
sub2['NUMBEERMO_EST'] = sub2['BEER_FEQMO'] * sub2['S2AQ5D']
sub2['NUMBEERMO_EST'] = pd.to_numeric(sub2['NUMBEERMO_EST'])



ct1 = sub2.groupby('NUMBEERMO_EST').size()
print(ct1)



sub2['DYSLIFE'] = sub2['DYSLIFE'].astype('category') 



%matplotlib inline
sns.boxplot(x='DYSLIFE', y='NUMBEERMO_EST', data=sub2)
plt.xlabel('DYSLIFE')
plt.ylabel('NUMBEERMO_EST')



model1 = smf.ols(formula='NUMBEERMO_EST ~ C(DYSLIFE)', data=sub2).fit()
print(model1.summary())



sub3 = sub2[['NUMBEERMO_EST', 'DYSLIFE']].dropna()



print('means for NUMBEERMO_EST by minor depression status')
m1 = sub3.groupby('DYSLIFE').mean()
print(m1)



print('standard deviations for NUMBEERMO_EST by minor depression status')
sd1 = sub3.groupby('DYSLIFE').std()
print(sd1)



sub2['ETHRACE2A'] = sub2['ETHRACE2A'].astype('category') 
sub2['ETHRACE2A'] = sub2['ETHRACE2A'].cat.rename_categories(["White", "Black", "NatAm", "Asian", "Hispanic"])



%matplotlib inline
sns.boxplot(x='ETHRACE2A', y='NUMBEERMO_EST', data=sub2)
plt.xlabel('ETHRACE2A')
plt.ylabel('NUMBEERMO_EST')



sub4 = sub2[['NUMBEERMO_EST', 'ETHRACE2A']].dropna()



model2 = smf.ols(formula='NUMBEERMO_EST ~ C(ETHRACE2A)', data=sub4).fit()
print(model2.summary())



print('means for NUMBEERMO_EST by ethinicity')
m2 = sub4.groupby('ETHRACE2A').mean()
print(m2)



print('standard deviations for NUMBEERMO_EST  by ethnicity')
sd2 = sub4.groupby('ETHRACE2A').std()
print(sd2)



mc1 = multi.MultiComparison(sub4['NUMBEERMO_EST'], sub4['ETHRACE2A'])
res1 = mc1.tukeyhsd()
print(res1.summary())