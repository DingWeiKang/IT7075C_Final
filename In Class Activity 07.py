import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df1 = pd.read_csv('Arthritis.csv')

# Dataset 1: Arthritis dataset
# Question 1: Draw a frequency plot with age of patients.
fig, ax = plt.subplots()
df1['Age'].value_counts().plot(ax=ax, kind='bar')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Question 1')
plt.show()
# Question 2: Draw a histogram with density curve with age of patients.
age=df1['Age']
sns.distplot(a=age, bins=40, color='purple',
             hist_kws={"edgecolor": 'black'})
plt.title('Question 2')
plt.show()
# Question 3: Draw a simple bar plot to show the frequency of treatment classes.
sns.countplot(x="Treatment",data=df1)
plt.title('Question 3')
plt.show()
# Question 4: Draw a stacked bar plot to show the frequency of improvement classes by treatment group?
df1.groupby(['Treatment','Improved']).sum().unstack().plot(kind='bar',y='Age',stacked=True,figsize=(6,8),ylabel='frequency',title='Question 4')
plt.show()

# Dataset 2: mtcars dataset
df2 = pd.read_csv('mtcars.csv')
# Question 5: Draw a box plot with mpg.
mpg=df2['mpg']
plt.boxplot(mpg)
plt.xlabel('mpg')
plt.ylabel('values')
plt.title('Question 5')
plt.show()
# Question 6: Draw a scatterplot with mpg across different car models.
x6=df2['model']
y6=df2['mpg']
plt.title('Question 6')
plt.scatter(x=y6,y=x6, c ="blue")
plt.show()
# Question 7: Draw a scatterplot with mpg for car models, grouped by cylinder
sns.scatterplot(x='mpg',y='model',data=df2,hue='cyl',legend=True).set(title='Question 7')
plt.show()
# Question 8: Fit a linear regression line using wt on x-axis and mpg on y-axis. (Hint: you can use sklearn package)
x8=list(df2['wt'])
y8=list(df2['mpg'])
plt.xlabel('wt')
plt.ylabel('mpg')
plt.title('Question 8')
sns.regplot(x8, y8, ci=None)
plt.show()
