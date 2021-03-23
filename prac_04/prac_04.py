import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



nesarc = pd.read_csv('nesarc.csv', low_memory=False)
pd.set_option('display.float_format', lambda x:'%f'%x)



nesarc['S2AQ5B'] = pd.to_numeric(nesarc['S2AQ5B'], errors='coerce')
nesarc['S2AQ5D'] = pd.to_numeric(nesarc['S2AQ5D'], errors='coerce')
nesarc['S2AQ5A'] = pd.to_numeric(nesarc['S2AQ5A'], errors='coerce')
nesarc['S2BQ1B1'] = pd.to_numeric(nesarc['S2BQ1B1'], errors='coerce')
nesarc['AGE'] = pd.to_numeric(nesarc['AGE'], errors='coerce')



sub1=nesarc[(nesarc['AGE']>=26) & (nesarc['AGE']<=50) & (nesarc['S2AQ5A']==1)]
sub2=sub1.copy()



sub2['S2AQ5D']=sub2['S2AQ5D'].replace(99, np.nan)

sub2['S2AQ5B']=sub2['S2AQ5B'].replace(8, np.nan)
sub2['S2AQ5B']=sub2['S2AQ5B'].replace(9, np.nan)
sub2['S2AQ5B']=sub2['S2AQ5B'].replace(10, np.nan)
sub2['S2AQ5B']=sub2['S2AQ5B'].replace(99, np.nan)

sub2['S2BQ1B1']=sub2['S2BQ1B1'].replace(9, np.nan)



recode2 = {1:30, 2:26, 3:14, 4:8, 5:4, 6:2.5, 7:1}
sub2['BEER_FEQMO']= sub2['S2AQ5B'].map(recode2)

recode3 = {2:0, 1:1}
sub2['S2BQ1B1']= sub2['S2BQ1B1'].map(recode3)



# A secondary variable multiplying the number of days beer consumed/month and the approx number of 
# beer consumed/day
sub2['NUMBEERMO_EST']=sub2['BEER_FEQMO'] * sub2['S2AQ5D']



var = sub2.groupby(['AGE']).NUMBEERMO_EST.mean()
print(var)



%matplotlib inline
#code for line chart
var.plot(kind="line")



var2 = sub2.groupby(['AGE']).NUMBEERMO_EST.sum()
print(var2)



fig = plt.figure()
# code for line chart
var2.plot(kind="line")



var3 = sub2.groupby(['AGE', 'S2BQ1B1']).NUMBEERMO_EST.mean()
print(var3)



# code for vertical stack chart
var3.unstack().plot(kind="bar", stacked=True, color=['red', 'blue'], grid=False)



# code for horizontal stack chart
var3.unstack().plot(kind="barh", stacked=True, color=['red', 'blue'], grid=False)



print(var2)



fig = plt.figure()

# code for pie chart
var2.plot(kind='pie',autopct='%.2f')



sub2['S1Q10A'] = pd.to_numeric(nesarc['S1Q10A'])



fig = plt.figure()

#code for violin chart
sns.violinplot(x='AGE', y='S1Q10A', data=sub2)

plt.xlabel('Age')
plt.ylabel('Income')



# you can rename categorical variable values for graphing if original values are not informative 
# first change the variable format to categorical if you haven’t already done so
sub2['ETHRACE2A'] = sub2['ETHRACE2A'].astype('category')

sub2['ETHRACE2A']=sub2['ETHRACE2A'].cat.rename_categories(["White", "Black", "NatAm", "Asian", "Hispanic"])



def CARTON_ADAY(row):
    if row['BEER_FEQMO'] >= 30:
        return 1
    elif row['BEER_FEQMO'] < 30:
        return 0
    
sub2['CARTON_ADAY'] = sub2.apply(lambda row: CARTON_ADAY(row), axis=1) 



c4= sub2.groupby('CARTON_ADAY').size()
print(c4)



# bivariate bar graph C->C
%matplotlib inline

#code for bar chart
sns.catplot(x='ETHRACE2A', y='CARTON_ADAY', data=sub2, kind="bar", ci=None)

plt.xlabel('Ethnic Group')
plt.ylabel('Proportion of consumed a carton a day Beer Drinkers')



sub3 = sub2[['ETHRACE2A','CARTON_ADAY']].copy()
sub3.head()



table = pd.pivot_table(sub3, index=['ETHRACE2A'], columns=['CARTON_ADAY'], aggfunc=np.size)
print(table)



fig = plt.figure()
# code for heat map
sns.heatmap(table)



pd.set_option('display.float_format', lambda x:'%.2f'%x)

gapminder = pd.read_csv('gapminder.csv', low_memory=False)
gapminder.head()



gapminder['internetuserate'] = pd.to_numeric(gapminder['internetuserate'], errors='coerce')
gapminder['urbanrate'] = pd.to_numeric(gapminder['urbanrate'], errors='coerce')
gapminder['incomeperperson'] = pd.to_numeric(gapminder['incomeperperson'], errors='coerce')



gapminder_clean=gapminder.dropna()



%matplotlib inline
fig = plt.figure()

#bubble plot code
plt.scatter(gapminder_clean['incomeperperson'], gapminder_clean['internetuserate'], s=gapminder_clean['urbanrate']) 

plt.xlabel('Ubran Rate')
plt.ylabel('Income Per Person')