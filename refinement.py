import numpy as np
import matplotlib.pyplot as plt
from exploration import df_salary
from scipy.stats import boxcox

# replacement of misspellings / other words
df_salary['Education Level'] = df_salary['Education Level'].replace({
    "Bachelor's Degree": "Bachelor's",
    "Master's Degree": "Master's",
    "phD": "PhD"
})

# review
print('\nNumber of entries for each education level:')
print(df_salary['Education Level'].value_counts())


# remove NA values
df_salary_cleaned = df_salary.dropna()

# review
print('\nNumber of NA values for each column:')
print(df_salary_cleaned.isnull().sum())


# remove duplicates --> not used in the following - only demonstration
df_salary_wo_duplicates = df_salary_cleaned.drop_duplicates()


# handling of outliers --> remove only salaries below 20.000
df_salary_cleaned = df_salary_cleaned[df_salary_cleaned['Salary'] >= 20000]

# review
print("\n Minimum salary:")
print(df_salary_cleaned['Salary'].min())


# remove illogical data
df_salary_cleaned = df_salary_cleaned[df_salary_cleaned['Career start (age)'] >= 18]

# review
print('\nDataFrame with illogical career start ages:')
print(df_salary_cleaned[(df_salary_cleaned['Career start (age)'] < 18)])


# Feature transformation - squared, logarithmic and box-cox transformation for salary
df_salary_cleaned['Squared'] = np.sqrt(df_salary_cleaned['Salary'])
df_salary_cleaned['Logarithmic'] = np.log(df_salary_cleaned['Salary'] + 1)
df_salary_cleaned['BoxCox'], _ = boxcox(df_salary_cleaned['Salary'] + 1)

plt.figure(figsize=(12, 8))

# Original data - histogram
plt.subplot(2, 2, 1)
plt.hist(df_salary_cleaned['Salary'], bins=6, color='skyblue', edgecolor='black')
plt.title('Original Salary Distribution', fontweight='bold')
plt.xlabel('Salary')
plt.ylabel('Frequency')

# Squared Transformation - histogram
plt.subplot(2, 2, 2)
plt.hist(df_salary_cleaned['Squared'], bins=6, color='lightgreen', edgecolor='black')
plt.title('Squared Transformation', fontweight='bold')
plt.xlabel('Squared Salary')
plt.ylabel('Frequency')

# Logarithmic Transformation - histogram
plt.subplot(2, 2, 3)
plt.hist(df_salary_cleaned['Logarithmic'], bins=6, color='salmon', edgecolor='black')
plt.title('Logarithmic Transformation', fontweight='bold')
plt.xlabel('Logarithmic Salary')
plt.ylabel('Frequency')

# Box-Cox-Transformation - histogram
plt.subplot(2, 2, 4)
plt.hist(df_salary_cleaned['BoxCox'], bins=6, color='orange', edgecolor='black')
plt.title('Box-Cox Transformation', fontweight='bold')
plt.xlabel('Box-Cox Transformed Salary')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# Feature construction
# adding of year of birth (data from 2024)
current_year = 2024
df_salary_cleaned['Birth Year'] = current_year - df_salary_cleaned['Age']

# adding of salary rank
df_salary_cleaned['Salary Rank'] = df_salary_cleaned['Salary'].rank(ascending=False, method='min')

# review
print('Data set "Salary data" including new columns:')
print(df_salary_cleaned)


# save cleaned file
df_salary_cleaned.to_csv(r'D:\Fortbildung\IU - Data Analyst Python\4. Modul - Data Quality and Data Wrangling\Task\Cleaned_Salary_Data.csv', index=False)
