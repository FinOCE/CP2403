import pandas as pd
import numpy as np
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt



#read in csv file into 
nesarc = pd.read_csv('nesarc.csv', low_memory=False) #increase efficiency
pd.set_option('display.float_format', lambda x:'%f'%x)



#setting variables you will be working with to numeric
nesarc['S2AQ5B'] = pd.to_numeric(nesarc['S2AQ5B'], errors='coerce') #convert variable to numeric
nesarc['S2AQ5D'] = pd.to_numeric(nesarc['S2AQ5D'], errors='coerce') #convert variable to numeric
nesarc['S2AQ5A'] = pd.to_numeric(nesarc['S2AQ5A'], errors='coerce') #convert variable to numeric
nesarc['S2BQ1B1'] = pd.to_numeric(nesarc['S2BQ1B1'], errors='coerce') #convert variable to numeric
nesarc['AGE'] = pd.to_numeric(nesarc['AGE'], errors='coerce') #convert variable to numeric



#subset data to adults age 26 to 50 who have consumed beer in the past 12 months
sub1 = nesarc[(nesarc['AGE'] >= 26) & (nesarc['AGE'] <= 50) & (nesarc['S2AQ5A'] == 1)]



sub2 = sub1.copy()



#SETTING MISSING DATA
sub2['S2AQ5D'] = sub2['S2AQ5D'].replace(99, np.nan)

sub2['S2AQ5B'] = sub2['S2AQ5B'].replace(8, np.nan)
sub2['S2AQ5B'] = sub2['S2AQ5B'].replace(9, np.nan)
sub2['S2AQ5B'] = sub2['S2AQ5B'].replace(10, np.nan)
sub2['S2AQ5B'] = sub2['S2AQ5B'].replace(99, np.nan)

sub2['S2BQ1B1']=sub2['S2BQ1B1'].replace(9, np.nan)



#recoding number of days consumed beer in the past month
recode2 = {1:30, 2:26, 3:14, 4:8, 5:4, 6:2.5, 7:1}
sub2['BEER_FEQMO'] = sub2['S2AQ5B'].map(recode2)

recode3 = {2:0, 1:1}
sub2['S2BQ1B1'] = sub2['S2BQ1B1'].map(recode3)



ct1 = pd.crosstab(sub2['S2BQ1B1'], sub2['BEER_FEQMO'])
print(ct1)



colsum = ct1.sum(axis=0)
colpct = ct1/colsum
print(colpct)



print('chi-square value, p value, expected counts')
cs1 = scipy.stats.chi2_contingency(ct1)
print(cs1)



%matplotlib inline
sns.factorplot(kind='bar', x='BEER_FEQMO', y='S2BQ1B1', data=sub2, ci=None)
plt.xlabel('Days drink beer per month')
plt.ylabel('Proportion Beer Dependent')



recode2 = {1: 1, 2.5: 2.5}
sub2['COMP1v2'] = sub2['BEER_FEQMO'].map(recode2)



# contingency table of observed counts
ct2 = pd.crosstab(sub2['S2BQ1B1'], sub2['COMP1v2'])
print(ct2)



# column percentages
colsum = ct2.sum(axis=0)
colpct = ct2/colsum
print(colpct)



print('chi-square value, p value, expected counts')
cs2 = scipy.stats.chi2_contingency(ct2)
print(cs2)



sub3 = sub2.copy()
cat = [1,2.5,4,8,14,26,30]

for x in range(0,len(cat)-1):
    for y in range(x+1,len(cat)):
        recode = {cat[x]: cat[x], cat[y]: cat[y]}
        
        sub3['temp'] = sub3['BEER_FEQMO'].map(recode)
        cont = pd.crosstab(sub3['S2BQ1B1'], sub3['temp'])
        
        cs = scipy.stats.chi2_contingency(cont)
        print('\n', cat[x], ' versus ', cat[y], 'Chi value: ', cs[0], '\tp value: ', cs[1])