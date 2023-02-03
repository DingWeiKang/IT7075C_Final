import pandas as pd

df = pd.read_csv('towed-vehicles.csv')

# Question 1
print("Question 1")
print(df.columns)

# Question 2
print("Question 2")
num_cols = df._get_numeric_data().columns
print(df.dtypes)
print(df._get_numeric_data().columns)

# Question 3
print("Question 3")


# Question 4
print("Question 4")
print(df)

# Question 5
print("Question 5")
print(df['Make'].value_counts())

# Question 6
print("Question 6")
print(df['Color'].value_counts())

# Question 7
print("Question 7")
pd.set_option('display.max_rows', None)
print(df['Plate'].value_counts())

# Question 8
print("Question 8")
print(((df['Color']=='BLK')&(df['Make']=='FORD')).value_counts())

# Question 9
print("Question 9")
print((df['State']!='OH').value_counts())