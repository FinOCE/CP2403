# Importing libraries
import pandas as pd
import numpy as np

# 1. Read in the nesarc.csv file
nesarc = pd.read_csv('nesarc.csv', low_memory=False)

# 2. Print the number of rows, columns in nesarc
print(len(nesarc))
print(len(nesarc.columns))

# Printing the first 5 rows of nesarc
print(nesarc.head(5))

# Convert Alcohol effects - 12 months (S2BQ1B1) to numeric & print first 10 rows
nesarc['S2BQ1B1'] = pd.to_numeric(nesarc['S2BQ1B1'], errors='coerce')
print(nesarc['S2BQ1B1'].head(10))

# Print the count and percentage of Alcohol effects - 12 months (S2BQ1B1)
print('counts for S2BQ1B1 alcohol effect in the past 12 months, yes=1')
c_al_dep = nesarc['S2BQ1B1'].value_counts(sort=False)
print(c_al_dep)

print('percentages for S2BQ1B1 alcohol effect in the past 12 months, yes=1')
p_al_dep = nesarc['S2BQ1B1'].value_counts(sort=False, normalize=True)
print(p_al_dep)

# Convert Beer drinking status (S2AQ5A) to numeric & print first 10 rows
nesarc['S2AQ5A'] = pd.to_numeric(nesarc['S2AQ5A'], errors='coerce')
nesarc['S2AQ5A'].head(25)

# Print the count and percentage of Beer drinking status (S2AQ5A)
c_beer_status = nesarc['S2AQ5A'].value_counts(sort=False, dropna=False)
print('counts for S2AQ5A beer drinking in the past year, yes=1')
print(c_beer_status)

p_beer_status = nesarc['S2AQ5A'].value_counts(sort=False, normalize=True, dropna=False)
print('percentages for S2AQ5A beer drinking in the past year, yes=1')
print(p_beer_status)

# Convert HOW OFTEN DRANK BEER IN LAST 12 MONTHS (S2AQ5B) to numeric & print first 10 rows
nesarc['S2AQ5B'] = pd.to_numeric(nesarc['S2AQ5B'], errors='coerce')
nesarc['S2AQ5B'].head(10)

# Print the count and percentage of HOW OFTEN DRANK BEER IN LAST 12 MONTHS (S2AQ5B)
nesarc['S2AQ5B'] = nesarc['S2AQ5B'].astype('category')

c_beer_feq = nesarc['S2AQ5B'].value_counts(sort=False, dropna=False)
print('counts for S2AQ5B – usual frequency when drinking beer')
print(c_beer_feq)

p_beer_feq = nesarc['S2AQ5B'].value_counts(sort=False, normalize=True, dropna=False)
print('percentages for S2AQ5B - usual frequency when drinking beer')
print(p_beer_feq)

# Convert NUMBER OF BEERS USUALLY CONSUMED ON DAYS WHEN DRANK BEER IN LAST 12 MONTHS (S2AQ5D) to numeric & print first 10 rows
nesarc['S2AQ5D'] = pd.to_numeric(nesarc['S2AQ5D'], errors='coerce')
nesarc['S2AQ5D'].head(10)

# Print the count and percentage of NUMBER OF BEERS USUALLY CONSUMED ON DAYS WHEN DRANK BEER IN LAST 12 MONTHS (S2AQ5D)
c_beer_quan = nesarc['S2AQ5D'].value_counts(sort=False, dropna=False)
print('counts for S2AQ5D usual quantity when drink beer')
print(c_beer_quan)

p_beer_quan = nesarc['S2AQ5D'].value_counts(sort=False, normalize=True, dropna=False)
print('percentages for S2AQ5D usual quantity when drink beer')
print(p_beer_quan)

# Use groupby() to calculate count & percentage for Alcohol effects - 12 months (S2BQ1B1)
c_al_dep_alt = nesarc.groupby('S2BQ1B1').size()
print(c_al_dep_alt)

p_al_dep_alt = nesarc.groupby('S2BQ1B1').size()*100/len(nesarc)
print(p_al_dep_alt)

# Obtain a subset of nesarc data, with the following criteria: Age from 26 to 50, Beer drinking status (S2AQ5A) = Y
nesarc['AGE'] = pd.to_numeric(nesarc['AGE'], errors='coerce')

sub1 = nesarc[(nesarc['AGE']>=26) & (nesarc['AGE']<=50) & (nesarc['S2AQ5A']==True)]
sub2 = sub1.copy()

c5 = sub2['AGE'].value_counts(sort=False)
print ('counts for AGE')
print(c5)

p5 = sub2['AGE'].value_counts(sort=False, normalize=True)
print ('percentages for AGE')
print (p5)