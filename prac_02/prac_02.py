# Set up
import pandas as pd
import numpy as np

nesarc = pd.read_csv('nesarc.csv', low_memory=False)
pd.set_option('display.float_format', lambda x:'%f'%x)

nesarc['S2AQ5A'] = pd.to_numeric(nesarc['S2AQ5A'], errors='coerce')
nesarc['S2AQ5B'] = pd.to_numeric(nesarc['S2AQ5B'], errors='coerce')
nesarc['S2AQ5D'] = pd.to_numeric(nesarc['S2AQ5D'], errors='coerce')

# For Beer drinking status (S2AQ5A) fill in nan value with 11 & print first 10 rows
nesarc['S2AQ5A'] = nesarc['S2AQ5A'].fillna(11)
nesarc['S2AQ5A'].head(10)

# For S2BQ1B1 - Effects of beer drinking (Beer Dependence) in the last 12 months replace 9 (unknown) in S2BQ1B1 (effects of beer consumtion in the last 12 months) to nan & print first 10 rows
nesarc['S2BQ1B1'] = nesarc['S2BQ1B1'].replace('9', np.NaN) # string '9' used since we didnt set to numeric earlier
nesarc['S2BQ1B1'].head(10)

# Recode S2BQ1B1 so that 0 is no, 1 is yes. currently 2 is no. & print first 5 rows
recode = nesarc['S2BQ1B1'].replace('2', '0') # string '2' and '0' used since we didnt set to numeric earlier
nesarc['S2BQ1B1'] = recode
nesarc['S2BQ1B1'].head(5)

# Obtain a subset of nesarc data, with the following criteria: Age from 26 to 50, Beer drinking status - S2AQ5A = Y
nesarc['AGE'] = pd.to_numeric(nesarc['AGE'])

# subset data to young adults age 26 to 50 who have drink beer in the past 12 months
sub1 = nesarc[(nesarc['AGE'] >= 26) & (nesarc['AGE'] <= 50) & (nesarc['S2AQ5A'] == 1)]

# Copy sub 1 to sub 2
sub2 = sub1.copy()
sub2.head()
len(sub2)

# Use sub2 data: Print the count of HOW OFTEN DRANK BEER IN LAST 12 MONTHS (S2AQ5B)
c_beer_feq = len(sub2['S2AQ5B'])
print('counts for original S2AQ5B')
print(c_beer_feq)

# Based on my research, I'm assuming that drinking less than once a month is not going to affect a person. So, we are going replace the following in 'HOW OFTEN DRANK BEER IN LAST 12 MONTHS (S2AQ5B)' to nan: 8, 9, 10, 99
sub2['S2AQ5B'] = sub2['S2AQ5B'].replace(8, np.NaN)
sub2['S2AQ5B'] = sub2['S2AQ5B'].replace(9, np.NaN)
sub2['S2AQ5B'] = sub2['S2AQ5B'].replace(10, np.NaN)
sub2['S2AQ5B'] = sub2['S2AQ5B'].replace(99, np.NaN)

# Use sub2 data: Print the count of HOW OFTEN DRANK BEER IN LAST 12 MONTHS (S2AQ5B) with 8, 9, 10 and 99 set nan
c_beer_feq_nan = len(sub2['S2AQ5B'].dropna())
print ('counts for original S2AQ5B with 8, 9, 10 and 99 set to NAN ')
print(c_beer_feq_nan)

# Use sub2 data: Count the NUMBER OF BEERS USUALLY CONSUMED ON DAYS WHEN DRANK BEER IN LAST 12 MONTHS (S2AQ5D)
c_beer_quan = len(sub2['S2AQ5D'])
print ('counts for S2AQ5D') 
print(c_beer_quan)

# Use sub2: Replace the 99 in 'NUMBER OF BEERS USUALLY CONSUMED ON DAYS WHEN DRANK BEER IN LAST 12 MONTHS (S2AQ5D)' to nan
sub2['S2AQ5D'] = sub2['S2AQ5D'].replace(99, np.NaN)

# Print the count of 'NUMBER OF BEERS USUALLY CONSUMED ON DAYS WHEN DRANK BEER IN LAST 12 MONTHS (S2AQ5D)'- with 99 set to NAN
c_beer_quan_nan = len(sub2['S2AQ5D'].dropna())
print ('counts for S2AQ5D with 99 set to NAN')
print(c_beer_quan_nan)

# Use sub2: Recode HOW OFTEN DRANK BEER IN LAST 12 MONTHS (S2AQ5B) as following: 1 to 7, 2 to 6, 3 to 5, 5 to 3, 6 to 2, 7 to 1, so that larger categorical numbers indicate more frequently someone drinks beer & print the count for BEER-FEQ
recode1 = sub2['S2AQ5B'].replace({1:7, 2:6, 3:5, 5:3, 6:2, 7:1}) #recoding so that higher numbers mean more smoking frequency
sub2['BEER_FEQ'] = recode1

recode_beer_feq = sub2['BEER_FEQ'].value_counts() #get count in each category
print('counts for S2AQ5B')
print(recode_beer_feq)

# Use sub 2: Recode HOW OFTEN DRANK BEER IN LAST 12 MONTHS (S2AQ5B) as following: 1 to 30, 2 to 26, 3 to 14, 4 to 8, 5 to 4, 6 to 2.5, 7 to 1, so that larger categorical numbers indicate more frequently someone drinks beer & print count of BEER_REQMO
#recoding values for S2AQ5B into a new variable, BEER_FEQMO
recode2 = sub2['S2AQ5B'].replace({1:30, 2:26, 3:14, 5:4, 6:2.5, 7:1}) #recode to quantitative variable
sub2['BEER_FEQMO'] = recode2

recode_beer_feq_m = sub2['BEER_FEQMO'].value_counts() #get count in each category
print ('counts for BEER_FEQMO')
print(recode_beer_feq_m)

# Use sub2: Create secondary variable NUMBEERMO_EST where NUMBEERMO_EST = BEER_FEQMO * S2AQ5D
sub2['NUMBEERMO_EST'] = sub2['BEER_FEQMO'] * sub2['S2AQ5D'] #get the number of beers consumed  per month
sub2['NUMBEERMO_EST'].head()

# print the count for age
#examining frequency distributions for age
c_age = sub2['AGE'].value_counts(sort=False)
print ('counts for AGE')
print(c_age)

# use sub2: print percentage for age
p_age = sub2['AGE'].value_counts(sort=False, normalize=True)
print ('percentages for AGE')
print (p_age)

# Group age into 3 groups: 26 - 33, 34 - 41, 42 - 50
sub2['AGEGROUP3'] = pd.cut(sub2.AGE, [25, 33, 41, 50])

# print the count of this new group
c_age_group = sub2['AGEGROUP3'].value_counts()
print('counts for AGEGROUP3')
print(c_age_group)

# print the percentage of this new group
print('percentages for AGEGROUP3')
p_age_group = sub2['AGEGROUP3'].value_counts(normalize=True)
print(p_age_group)

# Print the crosstab between AGEGROUP3 and AGE
# crosstabs evaluating which ages were put into which AGEGROUP3
print(pd.crosstab(sub2['AGEGROUP3'], sub2['AGE']))