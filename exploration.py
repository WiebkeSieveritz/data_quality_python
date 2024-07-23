import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from itertools import compress

# load csv file
file_path = r'D:\Fortbildung\IU - Data Analyst Python\4. Modul - Data Quality and Data Wrangling\Task\
Data set - Salary Data.csv'
df_salary = pd.read_csv(file_path)


# general information about the data set
print(df_salary.info())


# filtering for checking spelling
filtered_wo_duplicates = df_salary[['Gender', 'Education Level']].drop_duplicates()
print(filtered_wo_duplicates)


# pivot for interrecord structuring + checking missing values
pivot_education = pd.pivot_table(df_salary, values='Salary', index=['Gender'], columns=['Education Level'],
                                 aggfunc=np.sum)
print(pivot_education)


# check missing values
missing_values_per_column = df_salary.isnull().sum()
print("Missing values per column:")
print(missing_values_per_column)

empty_rows = df_salary.isnull().all(axis=1)
num_empty_rows = empty_rows.sum()
print(f"\nNumber of empty rows: {num_empty_rows}\n")


# check duplicates
duplicate_rows = df_salary[df_salary.duplicated(keep=False)]
unique_duplicate_rows = duplicate_rows.drop_duplicates()

num_unique_duplicates = unique_duplicate_rows.shape[0]
print(f"Number of duplicated rows: {num_unique_duplicates}")

# show duplicates
if num_unique_duplicates > 0:
    print("\nDuplicated rows:")
    print(unique_duplicate_rows)


# check outliers salary - z-score
mean_salary = np.mean(df_salary['Salary'])
std_salary = np.std(df_salary['Salary'])
data_z = (df_salary['Salary']-mean_salary)/std_salary
threshold = 3
print(list(compress(df_salary, data_z >= threshold)))

# check outliers salary - boxplot
Q1 = df_salary['Salary'].quantile(0.25)
Q3 = df_salary['Salary'].quantile(0.75)
IQR = Q3 - Q1

outliers = df_salary[(df_salary['Salary'] < (Q1 - 1.5 * IQR)) | (df_salary['Salary'] > (Q3 + 1.5 * IQR))]
print('outliers: ', outliers)

# check outliers salary - manually
print('Mean of salary: ', mean_salary)
print('Standard deviation of salary: ', std_salary)
outliers_manually = []
for x in df_salary['Salary']:
    if x <= 20000 or x >= (mean_salary + (2*std_salary)):
        outliers_manually.append(x)

print('\nOutliers in salary for manually fixed thresholds:')
print(outliers_manually)

# check outliers age
outliers_age = []
for x in df_salary['Age']:
    if x < 18 or x > 67:
        outliers_age.append(x)

print('\nOutliers in age <18 or >67:')
print(outliers_age)


# check illogical data
df_salary['Career start (age)'] = df_salary['Age']-df_salary['Years of Experience']
career_start = []
for x in df_salary['Career start (age)']:
    if x < 18 or x > 67:
        career_start.append(x)

print('\nIllogical start age:')
print(career_start)

df_illogical = df_salary[(df_salary['Career start (age)'] < 18) | (df_salary['Career start (age)'] > 67)]
print('\nDataFrame with illogical career start ages:')
print(df_illogical)


# normalization
scaler_norm = MinMaxScaler()
numeric_columns = ['Age', 'Years of Experience', 'Salary']
df_numeric = df_salary[numeric_columns]
df_normalization = scaler_norm.fit_transform(df_numeric)

df_normalized = pd.DataFrame(df_normalization, columns=numeric_columns)

# Visualization of normalized data - histograms for Age, Years of Experience and Salary
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(df_normalized['Age'], bins=20, color='blue', alpha=0.7)
plt.title('Normalized Age', fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(False)

plt.subplot(1, 3, 2)
plt.hist(df_normalized['Years of Experience'], bins=20, color='green', alpha=0.7)
plt.title('Normalized Years of Experience', fontweight='bold')
plt.xlabel('Years of Experience')
plt.ylabel('Frequency')
plt.grid(False)

plt.subplot(1, 3, 3)
plt.hist(df_normalized['Salary'], bins=20, color='red', alpha=0.7)
plt.title('Normalized Salary', fontweight='bold')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.grid(False)

plt.tight_layout(pad=1.0)
plt.show()


# standardization
scaler_std = StandardScaler()
df_standardization = scaler_std.fit_transform(df_numeric)

df_standardized = pd.DataFrame(df_standardization, columns=numeric_columns)

# Visualization of standardized data - histograms for Age, Years of Experience and Salary
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(df_standardized['Age'], bins=20, color='blue', alpha=0.7)
plt.title('Standardized Age', fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(False)

plt.subplot(1, 3, 2)
plt.hist(df_standardized['Years of Experience'], bins=20, color='green', alpha=0.7)
plt.title('Standardized Years of Experience', fontweight='bold')
plt.xlabel('Years of Experience')
plt.ylabel('Frequency')
plt.grid(False)

plt.subplot(1, 3, 3)
plt.hist(df_standardized['Salary'], bins=20, color='red', alpha=0.7)
plt.title('Standardized Salary', fontweight='bold')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.grid(False)

plt.tight_layout(pad=1.0)
plt.show()
