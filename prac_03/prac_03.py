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



sub2["S2BQ1B1"] = sub2["S2BQ1B1"].astype('category')



%matplotlib inline # using inline because notebook didn't seem to work with my IDE

# bar chart code here
sns.countplot(x="S2BQ1B1", data=sub2)

plt.xlabel('Beer Dependence past 12 months')
plt.title('Beer Dependence in the Past 12 Months Among Adult Drinkers'+ '\n' + ' in the NESARC Study')



sub2['NUMBEERMO_EST']=sub2['BEER_FEQMO'] * sub2['S2AQ5D']



%matplotlib inline

# Histogram plot code here
sns.histplot(sub2["NUMBEERMO_EST"].dropna(), kde=False) # using hitsplot instead of distplot due to deprecation warning

plt.xlabel('Number of Beers per Month')
plt.title('Estimated Number of Beers per Month' + '\n' + 'among Adult Beer Drinker in the NESARC Study')



print('describe number of beers drinking per month')
desc1 = sub2["NUMBEERMO_EST"].describe()
print(desc1)



print('mean')
mean1 = sub2["NUMBEERMO_EST"].mean()
print(mean1)

print('std')
std1 = sub2["NUMBEERMO_EST"].std()
print(std1)

print('min')
min1 = sub2["NUMBEERMO_EST"].min()
print(min1)

print ('max')
max1 = sub2["NUMBEERMO_EST"].max()
print(max1)

print ('median')
median1 = sub2["NUMBEERMO_EST"].median()
print(median1)

print ('mode')
mode1 = sub2["NUMBEERMO_EST"].mode()
print(mode1)



print('describe beer dependence')
desc2 = sub2['S2BQ1B1'].describe()
print(desc2)



sub2['S2BQ1B1'] = pd.to_numeric(sub2['S2BQ1B1']) # convert a numerical variable to quantitatie



print('describe beer dependence')
desc3 = sub2['S2BQ1B1'].describe()
print(desc3) #descriptor don't have sense



sub2['CARTONPERMONTH'] = sub2["NUMBEERMO_EST"] / 24



c2 = sub2.groupby('CARTONPERMONTH').size()
print(c2)



sub2['CARTONCATEGORY'] = pd.cut(sub2.CARTONPERMONTH, [0, 5, 10, 15, 20, 25, 30, np.inf])



# change format from numeric to categorical
sub2['CARTONCATEGORY'] = sub2['CARTONCATEGORY'].astype('category')



print('describe CARTONCATEGORY')
desc4 = sub2['CARTONCATEGORY'].describe()
print(desc4)



print('carton category counts')
c7 = sub2["CARTONCATEGORY"].value_counts(sort=False, dropna=True)
print(c7)



# bar chart code here
sns.catplot(x="CARTONCATEGORY", y="S2BQ1B1", data=sub2, kind="bar", ci=None) # changed factorplot to catplot due to rename

plt.xlabel('Carton per Month')
plt.ylabel('Proportion Beer Dependent')



# you can rename categorical variable values for graphing if original values are not informative 
# first change the variable format to categorical if you haven’t already done so
sub2['ETHRACE2A'] = sub2['ETHRACE2A'].astype('category')
# second create a new variable (PACKCAT) that has the new variable value labels
sub2['ETHRACE2A']=sub2['ETHRACE2A'].cat.rename_categories(["White", "Black", "NatAm", "Asian", "Hispanic"])



def CARTON_ADAY (row):
    if row['BEER_FEQMO'] >= 30 :
        return 1
    elif row['BEER_FEQMO'] < 30 :
        return 0
      
sub2['CARTON_ADAY'] = sub2.apply(lambda row: CARTON_ADAY(row),axis=1)
      
c4 = sub2.groupby('CARTON_ADAY').size()
print(c4)



# bar graph code here
sns.catplot(x="CARTON_ADAY", y="ETHRACE2A", data=sub2, kind="bar", ci=None)

plt.xlabel('Ethnic Group')
plt.ylabel('Proportion of drink beer a carton a day drinkers')



sub2['AGE'] = sub2['AGE'].astype('category') 
sub2['S1Q10A'] = pd.to_numeric(sub2['S1Q10A'])



%matplotlib inline

#box plot code here
sns.catplot(x="AGE", y="S1Q10A", data=sub2, kind="bar", ci=None)

plt.xlabel('Age')
plt.ylabel('Income')



pd.set_option('display.float_format', lambda x:'%.2f'%x)

gapminder = pd.read_csv('gapminder.csv', low_memory=False)
gapminder.head()



gapminder['oilperperson'] = pd.to_numeric(gapminder['oilperperson'], errors='coerce')
gapminder['relectricperperson'] = pd.to_numeric(gapminder['relectricperperson'], errors='coerce')



gapminder_clean = gapminder.dropna()



%matplotlib inline
plt.figure()

#scatter plot code here
scat = sns.regplot(x="relectricperperson", y="oilperperson", fit_reg=False, data=gapminder_clean)

plt.xlabel('Electrcity Use Per Person')
plt.ylabel('Oil Use Per Person')
plt.title('Scatterplot for the Association Between Electrcity Use Per Person' + '\n' + 'and Oil Use Per Person')