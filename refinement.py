import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from exploration import df_salary

# normalization
scaler_norm = MinMaxScaler()
numeric_columns = ['Age', 'Years of Experience', 'Salary']
df_salary[numeric_columns] = scaler_norm.fit_transform(df_salary[numeric_columns])

print(df_salary)

# Visualization of normalized data - histograms for Age, Years of Experience and Salary
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(df_salary['Age'], bins=20, color='blue', alpha=0.7)
plt.title('Normalized Age', fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(False)

plt.subplot(1, 3, 2)
plt.hist(df_salary['Years of Experience'], bins=20, color='green', alpha=0.7)
plt.title('Normalized Years of Experience', fontweight='bold')
plt.xlabel('Years of Experience')
plt.ylabel('Frequency')
plt.grid(False)

plt.subplot(1, 3, 3)
plt.hist(df_salary['Salary'], bins=20, color='red', alpha=0.7)
plt.title('Normalized Salary', fontweight='bold')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.grid(False)

plt.tight_layout(pad=1.0)
plt.show()


# standardization
scaler_std = StandardScaler()
numeric_columns = ['Age', 'Years of Experience', 'Salary']
df_salary[numeric_columns] = scaler_std.fit_transform(df_salary[numeric_columns])

print(df_salary)

# Visualization of standardized data - histograms for Age, Years of Experience and Salary
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(df_salary['Age'], bins=20, color='blue', alpha=0.7)
plt.title('Standardized Age', fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(False)

plt.subplot(1, 3, 2)
plt.hist(df_salary['Years of Experience'], bins=20, color='green', alpha=0.7)
plt.title('Standardized Years of Experience', fontweight='bold')
plt.xlabel('Years of Experience')
plt.ylabel('Frequency')
plt.grid(False)

plt.subplot(1, 3, 3)
plt.hist(df_salary['Salary'], bins=20, color='red', alpha=0.7)
plt.title('Standardized Salary', fontweight='bold')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.grid(False)

plt.tight_layout(pad=1.0)
plt.show()