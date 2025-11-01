Credit Card Users Churn Prediction
[ The Thera bank ]

preview.svg

Context
The Thera bank recently saw a steep decline in the number of users of their credit card, credit cards are a good source of income for banks because of different kinds of fees charged by the banks like annual fees, balance transfer fees, and cash advance fees, late payment fees, foreign transaction fees, and others. Some fees are charged to every user irrespective of usage, while others are charged under specified circumstances.

Objective
Customers’ leaving credit card services would lead the bank to loss, so the bank wants to analyze the data of customers and identify the customers who will leave their credit card services and the reason for same – so that the bank could improve upon those areas.

As a Data Scientist at Thera Bank need to explore the data provided, identify patterns, and come up with a classification model to identify customers likely to churn, and provide actionable insights and recommendations that will help the bank improve its services so that customers do not renounce their credit cards.

Data Description
CLIENTNUM: Client number. Unique identifier for the customer holding the account

Attrition_Flag: Internal event (customer activity) variable - if the account is closed then "Attrited Customer" else "Existing Customer"

Customer_Age: Age in Years

Gender: The gender of the account holder

Dependent_count: Number of dependents

Education_Level: Educational Qualification of the account holder - Graduate, High School, Unknown, Uneducated, College(refers to a college student), Post-Graduate, Doctorate.

Marital_Status: Marital Status of the account holder

Income_Category: Annual Income Category of the account holder

Card_Category: Type of Card

Months_on_book: Period of relationship with the bank

Total_Relationship_Count: Total no. of products held by the customer

Months_Inactive_12_mon: No. of months inactive in the last 12 months

Contacts_Count_12_mon: No. of Contacts between the customer and bank in the last 12 months

Credit_Limit: Credit Limit on the Credit Card

Total_Revolving_Bal: The balance that carries over from one month to the next is the revolving balance

Avg_Open_To_Buy: Open to Buy refers to the amount left on the credit card to use (Average of last 12 months)

Total_Trans_Amt: Total Transaction Amount (Last 12 months)

Total_Trans_Ct: Total Transaction Count (Last 12 months)

Total_Ct_Chng_Q4_Q1: Ratio of the total transaction count in 4th quarter and the total transaction count in the 1st quarter

Total_Amt_Chng_Q4_Q1: Ratio of the total transaction amount in 4th quarter and the total transaction amount in the 1st quarter

Avg_Utilization_Ratio: Represents how much of the available credit the customer spent

Libraries
# prompt: import necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
Data set
# import data set

df = pd.read_csv('/content/BankChurners.csv')
data=df.copy()
Analyzing the dataset
# Shape of the data set

data.shape
The dataset has 10127 rows ans 21 columns.
# Display the first five and last five rows rows of the DataFrame
print(data.head())
print ("--"*50)
print(data.tail())
# let's check for missing values in the data
round(data.isnull().sum() / data.isnull().count() * 100, 2)
The above table shows we have 2 columns with missing values
Education_Level - 1519
Marital_Status - 749
# duplicates in data
data.duplicated().sum()
The data set has no duplicate entries
# print uniques
data.nunique()
#  statistical summary of the numerical columns in the data
data.describe().T
Income has a max value of 666666 which is far greater than the mean and could be an outlier
Age has a large range of values
# the statistical summary of the non-numerical columns in the data
data.describe(exclude=np.number).T
Females, Graduate educated and married people are the majority of the data.
Data Cleaning
# Dropping column - ID
data.drop(columns=["CLIENTNUM"], inplace=True)
CLIENTNUM is dropped as it is unique for each customer and will not add value to the model.
# Making a list of all catrgorical variables
cat_col = data.select_dtypes("object").columns.to_list()

# Printing number of count of each unique value in each column
for column in cat_col:
    print(data[column].value_counts())
    print("-" * 40)
# Get value counts
df["Attrition_Flag"].value_counts(normalize=True)
We have more female customers as compared to male customers.

There are very few observations of Platinum Card holders (20).

# subset to view incorrect values
data[data.Income_Category == "abc"]
# replace values with missing
data.Income_Category.replace(to_replace="abc", value=np.nan, inplace=True)
# subset to view incorrect values
data[data.Income_Category == "abc"]
# impute mode onto missing values
data.Income_Category = data.Income_Category.fillna(
    value=data["Income_Category"].value_counts().index[0]
)
# check value replacement
data.Income_Category.value_counts()
Exploratory data analysis
Univariate Analysis
Attrition_Flag
# Calculate the value counts for 'Attrition_Flag'
attrition_counts = df['Attrition_Flag'].value_counts()

# Create the bar plot
plt.figure(figsize=(5, 5)) # size of the plot
sns.countplot(x='Attrition_Flag', data=df)
plt.title('Attrition Flag Distribution') # Title of the plot
plt.xlabel('Attrition Flag') # x-axis label
plt.ylabel('Count')

# Add the counts above each bar
for i, count in enumerate(attrition_counts):
    plt.text(i, count + 100, str(count), ha='center', va='bottom')  # Adjust vertical position

plt.show()
-> Insights

Majority of the customers are Existing Customers
Customer_Age
# analysiing using a boxplot

plt.figure(figsize=(6, 6))  # Set the figure size
sns.boxplot(x='Customer_Age', data=data , color='#FFCCB6')
plt.title('Customer Age Distribution ')  # Set the title
plt.xlabel('Attrition Flag')  # Set the x-axis label
plt.ylabel('Customer Age')  # Set the y-axis label
plt.show()
# analysiing using a histogram
plt.figure(figsize=(6, 6))  # Set the figure size
sns.histplot(data['Customer_Age'], kde=True, color='#AEC6CF')  # Create a histogram with a kernel density estimate
plt.title('Distribution of Customer Age')  # title
plt.xlabel('Customer Age') # title for x asix
plt.ylabel('Frequency') # title for y axis
plt.show()
-> Insights

The distribution of age is normal.
The boxplot shows that there are outliers at the right end
The outliers represent the real market trend so not trating them.
Gender
# Calculate the value counts for 'Gender'
gender_counts = data['Gender'].value_counts()

# Create the bar plot
plt.figure(figsize=(5, 5))  # Set the figure size
sns.countplot(x='Gender', data=data)
plt.title('Gender Distribution')  # Set the title
plt.xlabel('Gender')  # Set the x-axis label
plt.ylabel('Count')  # Set the y-axis label

# Add the counts above each bar
for i, count in enumerate(gender_counts):
    plt.text(i, count + 100, str(count), ha='center', va='bottom')  # Adjust vertical position

plt.show()
-> Insights

Female customers are taking more credit cards than male customers
There are approx 47% male customers and 53% are the female customers
Dependent_count
# analysiing using a boxplot
plt.figure(figsize=(8, 6))  # Set the figure size
sns.boxplot(x='Dependent_count', data=data, color='#FFCCB6')
plt.title('Dependent Count Distribution')  # Set the title
plt.xlabel('Dependent Count')  # Set the x-axis label
plt.ylabel('Count')  # Set the y-axis label
plt.show()

# analyzing using a histogram
plt.figure(figsize=(8, 6))  # Set the figure size
sns.histplot(data['Dependent_count'], kde=True, color='#AEC6CF')
plt.title('Distribution of Dependent Count')  # Set the title
plt.xlabel('Dependent Count')  # Set the x-axis label
plt.ylabel('Frequency')  # Set the y-axis label
plt.show()
-> Insights

Most customers have 2 or 3 dependents.
The distribution is skewed to the right.
Education_Level
# Calculate the value counts for 'Education_Level'
education_counts = data['Education_Level'].value_counts()

# Create the bar plot
plt.figure(figsize=(11, 5))  # Set the figure size
sns.countplot(x='Education_Level', data=data)
plt.title('Education Level Distribution')  # Set the title
plt.xlabel('Education Level')  # Set the x-axis label
plt.ylabel('Count')  # Set the y-axis label

# Add the counts above each bar
for i, count in enumerate(education_counts):
    plt.text(i, count + 100, str(count), ha='center', va='bottom')  # Adjust vertical position

plt.show()
-> Insights

Major of the customers, approx 31%, who take credit cards have their Graduate degree.
Approx 19% of customers are high school gradudates.
There are only 14% of customers who have no formal educaiton.
Marital_Status
# Calculate the value counts for 'Marital_Status'
marital_counts = data['Marital_Status'].value_counts()

# Create the bar plot
plt.figure(figsize=(8, 5))  # Set the figure size
sns.countplot(x='Marital_Status', data=data)
plt.title('Marital Status Distribution')  # Set the title
plt.xlabel('Marital Status')  # Set the x-axis label
plt.ylabel('Count')  # Set the y-axis label

# Add the counts above each bar
for i, count in enumerate(marital_counts):
    plt.text(i, count + 100, str(count), ha='center', va='bottom')  # Adjust vertical position

plt.show()
-> Insights

pprox 46% of customers are married. This makes sense as joint accounts are popular.
Approx 39% of customer are single.
low amount of divored customers
Income_Category
# Calculate the value counts for 'Income_Category'
income_counts = data['Income_Category'].value_counts()

# Create the bar plot
plt.figure(figsize=(10, 5))   # size of the plot
sns.countplot(x='Income_Category', data=data)
plt.title('Income Category Distribution')  # Title of the plot
plt.xlabel('Income Category')  # x-axis title
plt.ylabel('Count')  # y-axis title

# Add the counts above each bar
for i, count in enumerate(income_counts):
    plt.text(i, count + 100, str(count), ha='center', va='bottom')  # Adjust vertical position

plt.show()
-> Insights

The above plot shows the highest income and lowest income categories of the customers.
Card_Category
# Calculate the value counts for 'Card_Category'
card_category_counts = data['Card_Category'].value_counts()

# Create the bar plot
plt.figure(figsize=(8, 6))  # Set the figure size
sns.countplot(x='Card_Category', data=data)
plt.title('Card Category Distribution')  # Set the title
plt.xlabel('Card Category')  # Set the x-axis label
plt.ylabel('Count')  # Set the y-axis label

# Add the counts above each bar
for i, count in enumerate(card_category_counts):
    plt.text(i, count + 10, str(count), ha='center', va='bottom')  # Adjust vertical position

plt.show()
-> Insights

Majority of the customers i.e. 94% fall into the Blue category.
There are only approx 1% of customers that lie in the Gold category which makes sense as these may be the persons with high credit or high income.
There are very few observations, approx .2%, with Platinum category.
Months_on_book
# Histogram of Months_on_book
plt.figure(figsize=(10, 6)) # size of the plot
sns.histplot(data['Months_on_book'], kde=True, color='#AEC6CF')
plt.title('Distribution of Months on Book') # title of the plot
plt.xlabel('Months on Book') # X-axis title
plt.ylabel('Frequency')# y - axis title
plt.show()

# Boxplot of Months_on_book
plt.figure(figsize=(10, 6)) # size of the plot
sns.boxplot(x='Months_on_book', data=data, color='#FFCCB6')
plt.title('Months on Book Distribution') # title of the plot
plt.xlabel('Months on Book')
plt.show()
-> Insights

The distribution for the amount spent on fruits is highly normal.
The median lies around 37 months on the books for these customers at Thera Bank. There are some outliers on the right and left ends of the boxplot
 
### Total_Relationship_Count

# Boxplot of Total_Relationship_Count
plt.figure(figsize=(10, 5))  # size of the plot
sns.boxplot(x='Total_Relationship_Count', data=data, color='#FFCCB6')
plt.title('Total Relationship Count Distribution') # title of the plot
plt.xlabel('Total Relationship Count')# x-axis label
plt.show()

# Histogram of Total_Relationship_Count
plt.figure(figsize=(10, 5)) # size of the plot
sns.histplot(data['Total_Relationship_Count'], kde=True, color='#AEC6CF')
plt.title('Distribution of Total Relationship Count') # title of the plot
plt.xlabel('Total Relationship Count') # x- axis label
plt.ylabel('Frequency')
plt.show()
-> Insights

The distribution of Total_Relationship_Count is skewed to the left
It indicating that most customers have a relatively small number of relationships with the bank, while a few have a significantly higher number.
The boxplot shows that there are no significant outliers.
The most frequent relationship count is 3, followed by 4
Months_Inactive_12_mon
# Histogram of Months_Inactive_12_mon
plt.figure(figsize=(8, 5)) # size of the plot
sns.histplot(data['Months_Inactive_12_mon'], kde=True, color='#AEC6CF')
plt.title('Distribution of Months Inactive in the last 12 Months') # title of the plot
plt.xlabel('Months Inactive') # x-axis title
plt.ylabel('Frequency') # y-axis title
plt.show()

# Boxplot of Months_Inactive_12_mon
plt.figure(figsize=(8, 5)) # size of the plot
sns.boxplot(x='Months_Inactive_12_mon', data=data, color='#FFCCB6')
plt.title('Months Inactive in the last 12 Months Distribution') # title of the plot
plt.xlabel('Months Inactive')
plt.show()
-> Insights

The plot has outliers.
Contacts_Count_12_mon
# Calculate the value counts for 'Card_Category'
card_category_counts = data['Card_Category'].value_counts()

# Create the bar plot
plt.figure(figsize=(8, 6))  # Set the figure size
sns.barplot(x=card_category_counts.index, y=card_category_counts.values) # Use barplot for labeled bars
plt.title('Card Category Distribution')  # Set the title
plt.xlabel('Card Category')  # Set the x-axis label
plt.ylabel('Count')  # Set the y-axis label

# Add the counts above each bar
for i, count in enumerate(card_category_counts):
    plt.text(i, count + 10, str(count), ha='center', va='bottom')  # Adjust vertical position

plt.show()
-> Insights

The plot shows that most customers have had 3 No. of Contacts between the customer and bank in the last 12 months. Approximately just 51% of customers have had atleast 1.
Credit_Limit
# analysiing using a boxplot
plt.figure(figsize=(10, 5))  # Set the figure size
sns.boxplot(x='Credit_Limit', data=data, palette='Set3')
plt.title('Credit Limit Distribution')  # Set the title
plt.xlabel('Attrition Flag')  # Set the x-axis label
plt.ylabel('Credit Limit')  # Set the y-axis label
plt.show()
# analysiing using a histogram
plt.figure(figsize=(8, 5)) # sizee of the plot
sns.histplot(data['Credit_Limit'], kde=True, color='#AEC6CF')
plt.title('Distribution of Credit Limit') # title of the plot
plt.xlabel('Credit Limit') # x-axis title
plt.ylabel('Frequency') # y-axis title
plt.show()
-> Insights

The distribution of the credit amount is right-skewed
The boxplot shows that there are outliers at the right end
Total_Revolving_Bal
# Checking 10 largest values of amount of credit limit
data.Total_Revolving_Bal.nlargest(10)
# Histogram of Total_Revolving_Bal
plt.figure(figsize=(10, 6)) # size of the plot
sns.histplot(data['Total_Revolving_Bal'], kde=True, color='#AEC6CF')
plt.title('Distribution of Total Revolving Balance') # title of the plot
plt.xlabel('Total Revolving Balance') # x-axis label
plt.ylabel('Frequency') # y- axis label
plt.show()

# Boxplot of Total_Revolving_Bal
plt.figure(figsize=(10, 6)) # size of the plot
sns.boxplot(x='Total_Revolving_Bal', data=data, color='#FFCCB6')
plt.title('Total Revolving Balance Distribution') # title of the plot
plt.xlabel('Total Revolving Balance') # x-axis label
plt.show()
-> Insights

The boxplot shows that there are outliers at the left end
Avg_Open_To_Buy
# Histogram of Avg_Open_To_Buy
plt.figure(figsize=(10, 5)) # size of the plot
sns.histplot(data['Avg_Open_To_Buy'], kde=True, color='#AEC6CF')
plt.title('Distribution of Average Open to Buy') # title of the plot
plt.xlabel('Average Open to Buy') # x-axis title
plt.ylabel('Frequency') # y- axis title
plt.show()

# Boxplot of Avg_Open_To_Buy
plt.figure(figsize=(10, 5)) # size of the plot
sns.boxplot(x='Avg_Open_To_Buy', data=data, color='#FFCCB6')
plt.title('Average Open to Buy Distribution') # title of the plot
plt.xlabel('Average Open to Buy') # x-axis label
plt.show()
-> Insights

The distribution for the amount left on the credit card to use (Average of last 12 months) is right-skewed.
There is many observations to the right extreme which can be considered as an outliers.
Total_Amt_Chng_Q4_Q1
# Histogram of Total_Amt_Chng_Q4_Q1
plt.figure(figsize=(8,5)) # size of the plot
sns.histplot(data['Total_Amt_Chng_Q4_Q1'], kde=True, color='#AEC6CF')
plt.title('Distribution of Total Amount Change Q4-Q1') # title of the plot
plt.xlabel('Total Amount Change Q4-Q1') # x-axis title
plt.ylabel('Frequency') # y-axis title
plt.show()

# Boxplot of Total_Amt_Chng_Q4_Q1
plt.figure(figsize=(8,5)) # size of the plot
sns.boxplot(x='Total_Amt_Chng_Q4_Q1', data=data, color='#FFCCB6')
plt.title('Total Amount Change Q4-Q1 Distribution') # title of the plot
plt.xlabel('Total Amount Change Q4-Q1') # x-axis of the plot
plt.show()
-> Insights

The distribution is right skewed.
Total_Trans_Amt
# Boxplot of Total_Trans_Amt
plt.figure(figsize=(8,5)) # size of the plot
sns.boxplot(x='Total_Trans_Amt', data=data, color='#FFCCB6')
plt.title('Total Transaction Amount Distribution') # title of the plot
plt.xlabel('Total Transaction Amount') # x-axis label
plt.show()

# Histogram of Total_Trans_Amt
plt.figure(figsize=(8,5)) # size of the plot
sns.histplot(data['Total_Trans_Amt'], kde=True, color='#AEC6CF')
plt.title('Distribution of Total Transaction Amount') # title of the plot
plt.xlabel('Total Transaction Amount') # x-axis label
plt.ylabel('Frequency')
plt.show()
->Insights

The distribution for the Total Transaction Amount (Last 12 months) is right-skewed
There are many outliers in the amount spent on above 12500.
Total_Trans_Ct
# Histogram of Total_Trans_Ct
plt.figure(figsize=(8,5))# size of the plot
sns.histplot(data['Total_Trans_Ct'], kde=True, color='#AEC6CF')
plt.title('Distribution of Total Transaction Count')
plt.xlabel('Total Transaction Count')
plt.ylabel('Frequency')
plt.show()

# Boxplot of Total_Trans_Ct
plt.figure(figsize=(8,5))
sns.boxplot(x='Total_Trans_Ct', data=data, color='#FFCCB6')
plt.title('Total Transaction Count Distribution')
plt.xlabel('Total Transaction Count')
plt.show()
-> Insights

Majority of the customers ~65 transactions in the last 12 months.
There some extreme values in the far right end.
   
### Total_Ct_Chng_Q4_Q1

# Histogram of Total_Ct_Chng_Q4_Q1
plt.figure(figsize=(8,5)) # size of the plot
sns.histplot(data['Total_Ct_Chng_Q4_Q1'], kde=True, color='#AEC6CF')
plt.title('Distribution of Total Transaction Count Change (Q4/Q1)') # title of the plot
plt.xlabel('Total Transaction Count Change (Q4/Q1)')
plt.ylabel('Frequency')
plt.show()

# Boxplot of Total_Ct_Chng_Q4_Q1
plt.figure(figsize=(8,5)) # size of the plot
sns.boxplot(x='Total_Ct_Chng_Q4_Q1', data=data, color='#FFCCB6')
plt.title('Total Transaction Count Change (Q4/Q1) Distribution') # title of the plot
plt.xlabel('Total Transaction Count Change (Q4/Q1)')
plt.show()
-> Insights

The median of the distribution is ~.6 i.e. 50% of customers have ~.6 or less than ~.6 ratio of the total transaction count in 4th quarter and the total transaction count in 1st quarter.
  
### Avg_Utilization_Ratio

# Histogram of Avg_Utilization_Ratio
plt.figure(figsize=(8,5))# size of the plot
sns.histplot(data['Avg_Utilization_Ratio'], kde=True, color='#AEC6CF')
plt.title('Distribution of Average Utilization Ratio') # title of the plot
plt.xlabel('Average Utilization Ratio') # x-axis label
plt.ylabel('Frequency')
plt.show()

# Boxplot of Avg_Utilization_Ratio
plt.figure(figsize=(8,5)) # size of the plot
sns.boxplot(x='Avg_Utilization_Ratio', data=data, color='#FFCCB6')
plt.title('Average Utilization Ratio Distribution') # title of the plot
plt.xlabel('Average Utilization Ratio') # x- axis label
plt.show()
-> Insights

The distribution is right skewed.
There are no outliers in this variable
Bivariate analysis
# seaborn pairplot
sns.pairplot(data, hue="Attrition_Flag")
<seaborn.axisgrid.PairGrid at 0x7d70752d3be0>
No description has been provided for this image
-> Insights

There are overlaps i.e. no clear distinction in the distribution of variables for people who have attrited and did not attrit.
Attrition Flag and Customer age
# Analyzing using boxplot
plt.figure(figsize=(8,5)) # size of the plot
sns.boxplot(x='Attrition_Flag', y='Customer_Age', data=data, palette='Set3')
plt.title('Customer Age Distribution by Attrition Flag') # title of the plot
plt.xlabel('Attrition Flag')
plt.ylabel('Customer Age')
plt.show()
No description has been provided for this image
-> Insights

The median age of attritees is higher than the median age of non-attritees.
This shows that younger customers are more likely to stay as customers.
There are outliers in boxplots of only existing customers.
Attrition flag and credit limit
#  'Attrition_Flag' and 'Credit_Limit' are columns of boxplot
plt.figure(figsize=(8, 5)) # size of the plot
sns.boxplot(x='Attrition_Flag', y='Credit_Limit', data=data, palette='Set3')
plt.title('Credit Limit Distribution by Attrition Flag') # title of the plot
plt.xlabel('Attrition Flag') # x-axis label
plt.ylabel('Credit Limit')
plt.show()
No description has been provided for this image
-> Insights

the third quartile amount of existing customers is much more than the third quartile amount of attrited customers. This shows that customers with high credit card limits are more likely to stay on as customers.
There are outliers in boxplots of both class distributions
Attrition flag and total revolution bal
#  generate boxplot  for analyzing
plt.figure(figsize=(6, 5)) # size of the plot
sns.boxplot(x='Attrition_Flag', y='Total_Revolving_Bal', data=data, palette='Set3')
plt.title('Total Revolving Balance Distribution by Attrition Flag') # title of the plot
plt.xlabel('Attrition Flag') # x-axis label
plt.ylabel('Total Revolving Balance') # y-axis label
plt.show()
No description has been provided for this image
->Insights

the second and third quartile duration of existing customers is much more than the second and third quartile duration of attrited customers.
attrition flag and avg open to buy
# analyzing using the boxplot

# Assuming 'data' is your DataFrame and 'Dependent_count' is the column you want to analyze.
plt.figure(figsize=(8, 6)) # size of the plot
sns.boxplot(x="Attrition_Flag", y="Avg_Open_To_Buy", data=data, color='#FFCCB6')

plt.show()
No description has been provided for this image
-> Insights

he plot shows that customers with higher amount left on the credit card to use (Average of last 12 months) are more likely to be existing customers.
There are outliers in both of the distributions
Attrition flag and customer age
# generate  barplot for attrition and customer age

plt.figure(figsize=(10, 6))
sns.countplot(x='Customer_Age', hue='Attrition_Flag', data=data)
plt.title('Attrition by Customer Age')
plt.xlabel('Customer Age')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent labels from overlapping
plt.show()
No description has been provided for this image
-> Insights

Older customers tend to be existing customers. Younger people might be more prone to looking for new and better credit options.

40-year-olds take up the majority of the dataset.

43 - 48 year olds tended to attrite.

Attrition flag and gender
# generating bar plot for Attrition flag and gender

# Create the bar plot
plt.figure(figsize=(6, 5))  # Set the figure size
sns.countplot(x='Attrition_Flag', hue='Gender', data=data)
plt.title('Attrition by Gender')  # Set the title
plt.xlabel('Attrition Flag')  # Set the x-axis label
plt.ylabel('Count')  # Set the y-axis label
plt.show()
No description has been provided for this image
-> Insights

There is no significant difference concerning the ratio of gender amongst existing and attited customers.
Attrition flag and card category
#  generating barblot of Attrition flag and card category

# Assuming 'data' is your DataFrame and 'Attrition_Flag' and 'Card_Category' are columns.
plt.figure(figsize=(8, 6))
sns.countplot(x='Card_Category', hue='Attrition_Flag', data=data)
plt.title('Attrition by Card Category')
plt.xlabel('Card Category')
plt.ylabel('Number of Customers')
plt.show()
No description has been provided for this image
-> Insights

Customers owning a Silver or Blue card are less likely to attrit
Customers with Platinum or Gold cards are more likley to attrit.
Attrition flag and income category
#  barplot of  Attrition flag and income category

# Assuming 'data' is your DataFrame and 'Attrition_Flag' and 'Income_Category' are columns.
plt.figure(figsize=(10, 6))
sns.countplot(x='Income_Category', hue='Attrition_Flag', data=data)
plt.title('Attrition by Income Category')
plt.xlabel('Income Category')
plt.ylabel('Number of Customers')
plt.show()
No description has been provided for this image
-> Insights

The ratio amongst all income levels is roughly equal.
Rich customers are slightly less likely to attrit as compared to other customers.
Dependent count and attrition flag
# Dependent count and attrition flag generating a barplot

# Assuming 'data' is your DataFrame and 'Dependent_count' and 'Attrition_Flag' are columns.
plt.figure(figsize=(8, 6))
sns.countplot(x='Dependent_count', hue='Attrition_Flag', data=data)
plt.title('Attrition by Dependent Count')
plt.xlabel('Dependent Count')
plt.ylabel('Number of Customers')
plt.show()
No description has been provided for this image
-> Insights

The plot shows that there is little difference in the ratio of existing vs attrited customers per number of dependents. Customers with a 3 or 4 dependents have a slightly higher chance of attrition than others.
marital status and aattrition flag
#  marital status and attrition flag generate barplot

# Assuming 'data' is your DataFrame and 'Marital_Status' and 'Attrition_Flag' are columns.
plt.figure(figsize=(8, 6))
sns.countplot(x='Marital_Status', hue='Attrition_Flag', data=data)
plt.title('Attrition by Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Number of Customers')
plt.show()
No description has been provided for this image
-> Insights

Customers who who are single have a slightly higher attrition rate than non-singles.
Married people make up most attritees.
attrition flag and month on book
#  generate barplot of attrition flag and month on book

plt.figure(figsize=(10, 6))
sns.countplot(x='Months_on_book', hue='Attrition_Flag', data=data)
plt.title('Attrition by Month on Book')
plt.xlabel('Months on Book')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent labels from overlapping
plt.show()
No description has been provided for this image
attrition flag and and total relationship count
# generate barplot of attrition flag and and total relationship count

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data' is your DataFrame
plt.figure(figsize=(8, 6))
sns.countplot(x='Total_Relationship_Count', hue='Attrition_Flag', data=data)
plt.title('Attrition by Total Relationship Count')
plt.xlabel('Total Relationship Count')
plt.ylabel('Number of Customers')
plt.show()
No description has been provided for this image
attrition flag and and months inactive 12 mon
#  generate barplot of attrition flag and and months inactive 12 mon

# Assuming 'data' is your DataFrame
plt.figure(figsize=(8, 6))
sns.countplot(x='Months_Inactive_12_mon', hue='Attrition_Flag', data=data)
plt.title('Attrition by Months Inactive in the last 12 Months')
plt.xlabel('Months Inactive in the last 12 Months')
plt.ylabel('Number of Customers')
plt.show()
No description has been provided for this image
attrition flag and and contacts count 12 mon
# genrate barplot of attrition flag and and contacts count 12 mon

# Assuming 'data' is your DataFrame
plt.figure(figsize=(8, 6))
sns.countplot(x='Contacts_Count_12_mon', hue='Attrition_Flag', data=data)
plt.title('Attrition by Contacts Count in the last 12 Months')
plt.xlabel('Contacts Count in the last 12 Months')
plt.ylabel('Number of Customers')
plt.show()
No description has been provided for this image
attrition flag and total amt chng q4 q1
# generate barplot of attrition flag and total amt chng q4 q1

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.barplot(x='Attrition_Flag', y='Total_Amt_Chng_Q4_Q1', data=data)
plt.title('Attrition by Total Amount Change (Q4 over Q1)')
plt.xlabel('Attrition Flag')
plt.ylabel('Total Amount Change (Q4 over Q1)')
plt.show()
No description has been provided for this image
attrition flag and total trans amt
#  generate barplot of attrition flag and total trans amt

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data' is your DataFrame and it's already loaded
plt.figure(figsize=(8, 6))
sns.barplot(x='Attrition_Flag', y='Total_Trans_Amt', data=data)
plt.title('Attrition by Total Transaction Amount')
plt.xlabel('Attrition Flag')
plt.ylabel('Total Transaction Amount')
plt.show()
No description has been provided for this image
attrition flag and total trans ct
# generate barplot of attrition flag and total trans ct

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data' is your DataFrame and it's already loaded
plt.figure(figsize=(8, 6))
sns.barplot(x='Attrition_Flag', y='Total_Trans_Ct', data=data)
plt.title('Attrition by Total Transaction Count')
plt.xlabel('Attrition Flag')
plt.ylabel('Total Transaction Count')
plt.show()
No description has been provided for this image
attrition flag and total ct chng q4 q1
# generate barplot of the attrition flag and total ct chng q4 q1

# Assuming 'data' is your DataFrame and it's already loaded
plt.figure(figsize=(8, 6))
sns.barplot(x='Attrition_Flag', y='Total_Ct_Chng_Q4_Q1', data=data)
plt.title('Attrition by Total Count Change (Q4 over Q1)')
plt.xlabel('Attrition Flag')
plt.ylabel('Total Count Change (Q4 over Q1)')
plt.show()
No description has been provided for this image
attrition flag and avg utilization ratio
# generate barplot of attrition flag and avg utilization ratio

# Assuming 'data' is your DataFrame and it's already loaded
plt.figure(figsize=(8, 6))
sns.barplot(x='Attrition_Flag', y='Avg_Utilization_Ratio', data=data)
plt.title('Attrition by Average Utilization Ratio')
plt.xlabel('Attrition Flag')
plt.ylabel('Average Utilization Ratio')
plt.show()
No description has been provided for this image
data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10127 entries, 0 to 10126
Data columns (total 20 columns):
 #   Column                    Non-Null Count  Dtype  
---  ------                    --------------  -----  
 0   Attrition_Flag            10127 non-null  object 
 1   Customer_Age              10127 non-null  int64  
 2   Gender                    10127 non-null  object 
 3   Dependent_count           10127 non-null  int64  
 4   Education_Level           8608 non-null   object 
 5   Marital_Status            9378 non-null   object 
 6   Income_Category           10127 non-null  object 
 7   Card_Category             10127 non-null  object 
 8   Months_on_book            10127 non-null  int64  
 9   Total_Relationship_Count  10127 non-null  int64  
 10  Months_Inactive_12_mon    10127 non-null  int64  
 11  Contacts_Count_12_mon     10127 non-null  int64  
 12  Credit_Limit              10127 non-null  float64
 13  Total_Revolving_Bal       10127 non-null  int64  
 14  Avg_Open_To_Buy           10127 non-null  float64
 15  Total_Amt_Chng_Q4_Q1      10127 non-null  float64
 16  Total_Trans_Amt           10127 non-null  int64  
 17  Total_Trans_Ct            10127 non-null  int64  
 18  Total_Ct_Chng_Q4_Q1       10127 non-null  float64
 19  Avg_Utilization_Ratio     10127 non-null  float64
dtypes: float64(5), int64(9), object(6)
memory usage: 1.5+ MB
# plot correlation matrix excluding the object type without droping the columns

# Assuming 'data' is your DataFrame.
numeric_data = data.select_dtypes(include=np.number)
correlation_matrix = numeric_data.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numeric Features')
plt.show()
No description has been provided for this image
-> Insights

Total_Trans_Ct is highly correlated with Total_Trans_Amt which makes sense since both are related to the transactions of the credit card holder.
Months_on_book is related to Customer_Age which makes some sense since older customers would have more time on the banks books.
Avg_Utilization_Ratio has a correlation with Total_Revolving_Bal which makes sense since both are related to blaance.
Other variables have no significant correlation between them.
Total_Trans_Ct and Total_Trans_Amt
# pair plot of Total_Trans_Ct and Total_Trans_Amt

# Assuming 'data' is your DataFrame and it's already loaded
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Total_Trans_Ct', y='Total_Trans_Amt', data=data)
plt.title('Total Transaction Count vs. Total Transaction Amount')
plt.xlabel('Total Transaction Count')
plt.ylabel('Total Transaction Amount')
plt.show()
No description has been provided for this image
Months_on_book and customer age
# generate a pairplot of Months_on_book and customer age

# Assuming 'data' is your DataFrame and it's already loaded
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Months_on_book', y='Customer_Age', data=data)
plt.title('Months on Book vs. Customer Age')
plt.xlabel('Months on Book')
plt.ylabel('Customer Age')
plt.show()
No description has been provided for this image
Avg_Utilization_Ratio and Total_Revolving_Bal
# generate pair plot of Avg_Utilization_Ratio and Total_Revolving_Bal

# Assuming 'data' is your DataFrame and it's already loaded
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Avg_Utilization_Ratio', y='Total_Revolving_Bal', data=data)
plt.title('Average Utilization Ratio vs. Total Revolving Balance')
plt.xlabel('Average Utilization Ratio')
plt.ylabel('Total Revolving Balance')
plt.show()
No description has been provided for this image
Multivariate analysis
Customer_Age vs Total_Trans_Ct & Total_Trans_Amt

# prompt: pairplot of  Customer_Age vs Total_Trans_Ct & Total_Trans_Amt

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.pairplot(data=df, vars=['Customer_Age', 'Total_Trans_Ct', 'Total_Trans_Amt'], hue='Attrition_Flag')
plt.show()
<Figure size 1000x800 with 0 Axes>
No description has been provided for this image
compare the distributions of age with transaction count and amount while also distinguishing between "Attrited" and "Existing" customers.
Income_Category vs Credit_Limit & Avg_Utilization_Ratio

# boxplot of Income_Category vs Credit_Limit & Avg_Utilization_Ratio

plt.figure(figsize=(10, 8))
sns.boxplot(x='Income_Category', y='Credit_Limit', data=df, hue='Attrition_Flag')
plt.title('Income Category vs Credit Limit')
plt.show()

plt.figure(figsize=(10, 8))
sns.boxplot(x='Income_Category', y='Avg_Utilization_Ratio', data=df, hue='Attrition_Flag')
plt.title('Income Category vs Average Utilization Ratio')
plt.show()
No description has been provided for this image
No description has been provided for this image
These plots will show how income categories correlate with credit limit and utilization, and how they affect customers' credit behavior.
Income_Category vs Credit_Limit & Avg_Utilization_Ratio

plt.figure(figsize=(10, 8))
sns.boxplot(x='Income_Category', y='Credit_Limit', data=df, hue='Attrition_Flag')
plt.title('Income Category vs Credit Limit')
plt.show()

plt.figure(figsize=(10, 8))
sns.boxplot(x='Income_Category', y='Avg_Utilization_Ratio', data=df, hue='Attrition_Flag')
plt.title('Income Category vs Average Utilization Ratio')
plt.show()
No description has been provided for this image
No description has been provided for this image
income categories correlate with credit limit and utilization, and how they affect customers' credit behavior.
. Months_on_book vs Total_Relationship_Count & Credit_Limit

plt.figure(figsize=(6, 6))
sns.heatmap(df[['Months_on_book', 'Total_Relationship_Count', 'Credit_Limit']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap of Months_on_book vs Total_Relationship_Count & Credit_Limit')
plt.show()
No description has been provided for this image
correlations between months on book, relationship count, and credit limit, helping identify how these variables interact.
Total_Relationship_Count vs Months_Inactive_12_mon & Contacts_Count_12_mon

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Total_Relationship_Count', y='Months_Inactive_12_mon', hue='Contacts_Count_12_mon', data=df)
plt.title('Total Relationship Count vs Months Inactive (12 Months) & Contacts Count (12 Months)')
plt.xlabel('Total Relationship Count')
plt.ylabel('Months Inactive (12 Months)')
plt.legend(title='Contacts Count (12 Months)', loc='upper right')
plt.show()
No description has been provided for this image
show how the number of products (Total_Relationship_Count) correlates with inactivity and customer contact, with color coding for attrition status.
Gender vs Avg_Open_To_Buy & Credit_Limit

# boxplot of Gender vs Avg_Open_To_Buy & Credit_Limit data=df

plt.figure(figsize=(10, 6))
sns.boxplot(x='Gender', y='Avg_Open_To_Buy', data=df)
plt.title('Gender vs. Average Open to Buy')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Gender', y='Credit_Limit', data=df)
plt.title('Gender vs. Credit Limit')
plt.show()
No description has been provided for this image
No description has been provided for this image
A boxplot will help compare gender against available credit (Avg_Open_To_Buy) and the credit limit, showing the distribution across genders.
Education_Level vs Income_Category & Card_Category

sns.countplot(x='Education_Level', hue='Income_Category', data=df)
plt.title('Education Level vs. Income Category')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x='Education_Level', hue='Card_Category', data=df)
plt.title('Education Level vs. Card Category')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
No description has been provided for this image
No description has been provided for this image
different income categories and education levels align with different types of cards, revealing insights into customer preferences.
Data Preparation for Modeling
print(data['Attrition_Flag'].unique())
['Existing Customer' 'Attrited Customer']
# label encode the target variable
data.Attrition_Flag = data.Attrition_Flag.replace(
    to_replace={"Attrited Customer": 1, "Existing Customer": 0}
)

data.Attrition_Flag
Attrition_Flag
0	0
1	0
2	0
3	0
4	0
...	...
10122	0
10123	1
10124	1
10125	1
10126	1
10127 rows × 1 columns


dtype: int64
# print head
data.head()
Attrition_Flag	Customer_Age	Gender	Dependent_count	Education_Level	Marital_Status	Income_Category	Card_Category	Months_on_book	Total_Relationship_Count	Months_Inactive_12_mon	Contacts_Count_12_mon	Credit_Limit	Total_Revolving_Bal	Avg_Open_To_Buy	Total_Amt_Chng_Q4_Q1	Total_Trans_Amt	Total_Trans_Ct	Total_Ct_Chng_Q4_Q1	Avg_Utilization_Ratio
0	0	45	M	3	High School	Married	
60
K
−
80K	Blue	39	5	1	3	12691.0	777	11914.0	1.335	1144	42	1.625	0.061
1	0	49	F	5	Graduate	Single	Less than $40K	Blue	44	6	1	2	8256.0	864	7392.0	1.541	1291	33	3.714	0.105
2	0	51	M	3	Graduate	Married	
80
K
−
120K	Blue	36	4	1	0	3418.0	0	3418.0	2.594	1887	20	2.333	0.000
3	0	40	F	4	High School	NaN	Less than $40K	Blue	34	3	4	1	3313.0	2517	796.0	1.405	1171	20	2.333	0.760
4	0	40	M	3	Uneducated	Married	
60
K
−
80K	Blue	21	5	1	0	4716.0	0	4716.0	2.175	816	28	2.500	0.000
# Separating target variable and other variables
X = data.drop(columns="Attrition_Flag")

# make dependent variable
Y = data["Attrition_Flag"]

# check head
X.head()
Customer_Age	Gender	Dependent_count	Education_Level	Marital_Status	Income_Category	Card_Category	Months_on_book	Total_Relationship_Count	Months_Inactive_12_mon	Contacts_Count_12_mon	Credit_Limit	Total_Revolving_Bal	Avg_Open_To_Buy	Total_Amt_Chng_Q4_Q1	Total_Trans_Amt	Total_Trans_Ct	Total_Ct_Chng_Q4_Q1	Avg_Utilization_Ratio
0	45	M	3	High School	Married	
60
K
−
80K	Blue	39	5	1	3	12691.0	777	11914.0	1.335	1144	42	1.625	0.061
1	49	F	5	Graduate	Single	Less than $40K	Blue	44	6	1	2	8256.0	864	7392.0	1.541	1291	33	3.714	0.105
2	51	M	3	Graduate	Married	
80
K
−
120K	Blue	36	4	1	0	3418.0	0	3418.0	2.594	1887	20	2.333	0.000
3	40	F	4	High School	NaN	Less than $40K	Blue	34	3	4	1	3313.0	2517	796.0	1.405	1171	20	2.333	0.760
4	40	M	3	Uneducated	Married	
60
K
−
80K	Blue	21	5	1	0	4716.0	0	4716.0	2.175	816	28	2.500	0.000
# set dummy variables
X = pd.get_dummies(
    data=data,
    columns=[
        "Gender",
        "Education_Level",  # has missing values
        "Marital_Status",  # has missing values
        "Income_Category",
        "Card_Category",
    ],
    drop_first=True,
)
# drop target variable
X = X.drop(columns="Attrition_Flag")
# check head
X.head()
Customer_Age	Dependent_count	Months_on_book	Total_Relationship_Count	Months_Inactive_12_mon	Contacts_Count_12_mon	Credit_Limit	Total_Revolving_Bal	Avg_Open_To_Buy	Total_Amt_Chng_Q4_Q1	...	Education_Level_Uneducated	Marital_Status_Married	Marital_Status_Single	Income_Category_
40
K
−
60K	Income_Category_
60
K
−
80K	Income_Category_
80
K
−
120K	Income_Category_Less than $40K	Card_Category_Gold	Card_Category_Platinum	Card_Category_Silver
0	45	3	39	5	1	3	12691.0	777	11914.0	1.335	...	False	True	False	False	True	False	False	False	False	False
1	49	5	44	6	1	2	8256.0	864	7392.0	1.541	...	False	False	True	False	False	False	True	False	False	False
2	51	3	36	4	1	0	3418.0	0	3418.0	2.594	...	False	True	False	False	False	True	False	False	False	False
3	40	4	34	3	4	1	3313.0	2517	796.0	1.405	...	False	False	False	False	False	False	True	False	False	False
4	40	3	21	5	1	0	4716.0	0	4716.0	2.175	...	True	True	False	False	True	False	False	False	False	False
5 rows × 29 columns

# print percentage of each unique value in the Y series
Y.value_counts() / 100
count
Attrition_Flag	
0	85.00
1	16.27

dtype: float64
Splitting data

# first we split data into 2 parts, say temporary and test

X_temp, X_test, y_temp, y_test = train_test_split(
    X, Y, test_size=0.20, random_state=1, stratify=Y
)

# then we split the temporary set into train and validation

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=1,
    stratify=y_temp,  # set the weighting feature on
)
print(X_train.shape, X_val.shape, X_test.shape)
(6075, 29) (2026, 29) (2026, 29)
# check head
X_train.head()
Customer_Age	Dependent_count	Months_on_book	Total_Relationship_Count	Months_Inactive_12_mon	Contacts_Count_12_mon	Credit_Limit	Total_Revolving_Bal	Avg_Open_To_Buy	Total_Amt_Chng_Q4_Q1	...	Education_Level_Uneducated	Marital_Status_Married	Marital_Status_Single	Income_Category_
40
K
−
60K	Income_Category_
60
K
−
80K	Income_Category_
80
K
−
120K	Income_Category_Less than $40K	Card_Category_Gold	Card_Category_Platinum	Card_Category_Silver
800	40	2	21	6	4	3	20056.0	1602	18454.0	0.466	...	False	False	True	False	False	False	False	False	False	False
498	44	1	34	6	2	0	2885.0	1895	990.0	0.387	...	False	True	False	False	False	False	True	False	False	False
4356	48	4	36	5	1	2	6798.0	2517	4281.0	0.873	...	False	True	False	False	False	True	False	False	False	False
407	41	2	36	6	2	0	27000.0	0	27000.0	0.610	...	False	False	False	False	True	False	False	False	False	True
8728	46	4	36	2	2	3	15034.0	1356	13678.0	0.754	...	False	False	False	True	False	False	False	False	False	True
5 rows × 29 columns

Missing-Value Treatment -> use median to impute missing values in Education_Level, Marital_Status.

# show missing values
data.isna().sum()
0
Attrition_Flag	0
Customer_Age	0
Gender	0
Dependent_count	0
Education_Level	1519
Marital_Status	749
Income_Category	0
Card_Category	0
Months_on_book	0
Total_Relationship_Count	0
Months_Inactive_12_mon	0
Contacts_Count_12_mon	0
Credit_Limit	0
Total_Revolving_Bal	0
Avg_Open_To_Buy	0
Total_Amt_Chng_Q4_Q1	0
Total_Trans_Amt	0
Total_Trans_Ct	0
Total_Ct_Chng_Q4_Q1	0
Avg_Utilization_Ratio	0

dtype: int64
# view missing data for type
data[data.Education_Level.isnull()]
Attrition_Flag	Customer_Age	Gender	Dependent_count	Education_Level	Marital_Status	Income_Category	Card_Category	Months_on_book	Total_Relationship_Count	Months_Inactive_12_mon	Contacts_Count_12_mon	Credit_Limit	Total_Revolving_Bal	Avg_Open_To_Buy	Total_Amt_Chng_Q4_Q1	Total_Trans_Amt	Total_Trans_Ct	Total_Ct_Chng_Q4_Q1	Avg_Utilization_Ratio
6	0	51	M	4	NaN	Married	$120K +	Gold	46	6	1	3	34516.0	2264	32252.0	1.975	1330	31	0.722	0.066
11	0	65	M	1	NaN	Married	
40
K
−
60K	Blue	54	6	2	3	9095.0	1587	7508.0	1.433	1314	26	1.364	0.174
15	0	44	M	4	NaN	NaN	
80
K
−
120K	Blue	37	5	1	2	4234.0	972	3262.0	1.707	1348	27	1.700	0.230
17	0	41	M	3	NaN	Married	
80
K
−
120K	Blue	34	4	4	1	13535.0	1291	12244.0	0.653	1028	21	1.625	0.095
23	0	47	F	4	NaN	Single	Less than $40K	Blue	36	3	3	2	2492.0	1560	932.0	0.573	1126	23	0.353	0.626
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
10090	0	36	F	3	NaN	Married	
40
K
−
60K	Blue	22	5	3	3	12958.0	2273	10685.0	0.608	15681	96	0.627	0.175
10094	0	59	M	1	NaN	Single	
60
K
−
80K	Blue	48	3	1	2	7288.0	0	7288.0	0.640	14873	120	0.714	0.000
10095	0	46	M	3	NaN	Married	
80
K
−
120K	Blue	33	4	1	3	34516.0	1099	33417.0	0.816	15490	110	0.618	0.032
10118	1	50	M	1	NaN	NaN	
80
K
−
120K	Blue	36	6	3	4	9959.0	952	9007.0	0.825	10310	63	1.100	0.096
10123	1	41	M	2	NaN	Divorced	
40
K
−
60K	Blue	25	4	2	3	4277.0	2186	2091.0	0.804	8764	69	0.683	0.511
1519 rows × 20 columns

from sklearn.impute import SimpleImputer
import numpy as np
# Let's impute the missing values
imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
impute = imputer.fit(X_train)

# impute on train, validation set and test set
X_train = impute.transform(X_train)
X_val = imputer.transform(X_val)
X_test = imputer.transform(X_test)
print(X_train)
print(X_val)
print(X_test)
[[40.  2. 21. ...  0.  0.  0.]
 [44.  1. 34. ...  0.  0.  0.]
 [48.  4. 36. ...  0.  0.  0.]
 ...
 [50.  0. 36. ...  0.  0.  0.]
 [45.  4. 38. ...  0.  0.  0.]
 [55.  2. 41. ...  0.  0.  0.]]
[[37.  0. 27. ...  0.  0.  0.]
 [58.  2. 46. ...  0.  0.  0.]
 [42.  3. 23. ...  0.  1.  0.]
 ...
 [45.  4. 35. ...  0.  0.  0.]
 [53.  2. 42. ...  0.  0.  0.]
 [38.  0. 32. ...  0.  0.  0.]]
[[32.  1. 26. ...  0.  0.  0.]
 [50.  1. 36. ...  0.  0.  0.]
 [54.  2. 36. ...  0.  0.  0.]
 ...
 [45.  2. 40. ...  0.  0.  0.]
 [33.  4. 26. ...  0.  0.  0.]
 [55.  2. 37. ...  0.  0.  0.]]
# check for missingness in training set
np.nan in X_train
False
# get column count
len(X_train.T)
29
Model building
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score
# Empty list to store all the models
models = []

# Appending models into the list
models.append(
    (
        "LogisticRegression",
        LogisticRegression(random_state=1, class_weight={0: 15, 1: 85}),
    )
)
models.append(
    (
        "DecisionTree",
        DecisionTreeClassifier(random_state=1, class_weight={0: 15, 1: 85}),
    )
)
models.append(("GBM", GradientBoostingClassifier(random_state=1)))
models.append(("Adaboost", AdaBoostClassifier(random_state=1)))
models.append(("Xgboost", XGBClassifier(random_state=1, eval_metric="logloss"),))
models.append(("Bagging", BaggingClassifier(random_state=1)))

results = []  # Empty list to store all model's CV scores
names = []  # Empty list to store name of the models


# loop through all models to get the mean cross validated score
print("\n" "Cross-Validation Performance:" "\n")

for name, model in models:
    scoring = "recall"
    kfold = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=1
    )  # Setting number of splits equal to 5
    cv_result = cross_val_score(
        estimator=model, X=X_train, y=y_train, scoring=scoring, cv=kfold
    )
    results.append(cv_result)
    names.append(name)
    print("{}: {}".format(name, cv_result.mean() * 100))

print("\n" "Validation Performance:" "\n")

for name, model in models:
    model.fit(X_train, y_train)
    scores = recall_score(y_val, model.predict(X_val))
    print("{}: {}".format(name, scores))
Cross-Validation Performance:

LogisticRegression: 80.94191522762951
DecisionTree: 74.6902145473574
GBM: 81.24646781789639
Adaboost: 81.3469387755102
Xgboost: 86.26844583987442
Bagging: 78.48142333856619

Validation Performance:

LogisticRegression: 0.8098159509202454
DecisionTree: 0.803680981595092
GBM: 0.8588957055214724
Adaboost: 0.8588957055214724
Xgboost: 0.8834355828220859
Bagging: 0.7975460122699386
# Plotting boxplots for CV scores of all models defined above
fig = plt.figure(figsize=(10, 7))

fig.suptitle("CV Comparison")
ax = fig.add_subplot(111)

plt.boxplot(results)
ax.set_xticklabels(names)

plt.show()
No description has been provided for this image
XGBoost is giving the highest cross-validated recall followed by Ada/GradientBoost.
The boxplot shows that the performance of AdaBoost and GradientBoost are consistent.
The Performance of Adaboost and XGBoost is generalised on validation set as well.
Hyperparameter Tuning
# defining a function to compute different metrics to check performance of a classification model built using sklearn
def model_performance_classification_sklearn(model, predictors, target):
    """
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    """

    # predicting using the independent variables
    pred = model.predict(predictors)

    acc = accuracy_score(target, pred)  # to compute Accuracy
    recall = recall_score(target, pred)  # to compute Recall
    precision = precision_score(target, pred)  # to compute Precision
    f1 = f1_score(target, pred)  # to compute F1-score

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {"Accuracy": acc, "Recall": recall, "Precision": precision, "F1": f1,},
        index=[0],
    )

    return df_perf
def confusion_matrix_sklearn(model, predictors, target):
    """
    To plot the confusion_matrix with percentages

    model: classifier
    predictors: independent variables
    target: dependent variable
    """
    y_pred = model.predict(predictors)
    cm = confusion_matrix(target, y_pred)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)

    # plot the matrix inside a heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
GridSearchCV

from sklearn import metrics
from sklearn.model_selection import GridSearchCV
# defining model
model = GradientBoostingClassifier(random_state=1)

# Parameter grid to pass in GridSearchCV
param_grid = {
    "n_estimators": np.arange(10, 110, 10),
    "learning_rate": [0.1, 0.01, 0.2, 0.05, 1]
}

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.recall_score)

# Calling GridSearchCV
grid_cv = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scorer, cv=5, n_jobs = -1)

# Fitting parameters in GridSeachCV
grid_cv.fit(X_train, y_train)

print(
    "Best Parameters:{} \nScore: {}".format(grid_cv.best_params_, grid_cv.best_score_)
)
Best Parameters:{'learning_rate': 0.2, 'n_estimators': 100} 
Score: 0.845264259549974
# extract best estimator from the model
gbgscv = grid_cv.best_estimator_
# fit the model
gbgscv.fit(X_train, y_train)

  GradientBoostingClassifier?i
GradientBoostingClassifier(learning_rate=0.2, random_state=1)
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# Calculating different metrics on train set
print("Training performance:")
gradient_grid_train = model_performance_classification_sklearn(gbgscv, X_train, y_train)
display(gradient_grid_train)
Training performance:
Accuracy	Recall	Precision	F1
0	0.986337	0.940574	0.973489	0.956748
# creating confusion matrix
confusion_matrix_sklearn(gbgscv, X_train, y_train)
No description has been provided for this image
# Calculating different metrics on validation set
print("Validation performance:")
gradient_grid_val = model_performance_classification_sklearn(gbgscv, X_val, y_val)
display(gradient_grid_val)
Validation performance:
Accuracy	Recall	Precision	F1
0	0.973346	0.895706	0.935897	0.915361
# creating confusion matrix
confusion_matrix_sklearn(gbgscv, X_val, y_val)
No description has been provided for this image
The tuned GradientBoost model is not overfitting the training data.
RandomizedSearchCV

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_score, f1_score
import numpy as np
# defining model
model = GradientBoostingClassifier(random_state=1)

# Parameter grid to pass in GridSearchCV
param_grid = {
    "n_estimators": np.arange(10, 110, 10),
    "learning_rate": [0.1, 0.01, 0.2, 0.05, 1]
}

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.recall_score)

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_jobs = -1, n_iter=50, scoring=scorer, cv=3, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train,y_train)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))
Best parameters are {'n_estimators': 80, 'learning_rate': 0.2} with CV score=0.8503948403334906:
# extract best estimator from the model
gbrs = randomized_cv.best_estimator_
# Fit the model on training data
gbrs.fit(X_train, y_train)

  GradientBoostingClassifier?i
GradientBoostingClassifier(learning_rate=0.2, n_estimators=80, random_state=1)
# Calculating different metrics on train set
print("Training performance:")
gradient_random_train = model_performance_classification_sklearn(gbrs, X_train, y_train)
display(gradient_random_train)
Training performance:
Accuracy	Recall	Precision	F1
0	0.981893	0.929303	0.956751	0.942827
# creating confusion matrix
confusion_matrix_sklearn(gbrs, X_train, y_train)
No description has been provided for this image
# Calculating different metrics on validation set
print("Validation performance:")
gradient_random_val = model_performance_classification_sklearn(gbrs, X_val, y_val)
display(gradient_random_val)
Validation performance:
Accuracy	Recall	Precision	F1
0	0.972359	0.883436	0.941176	0.911392
# creating confusion matrix
confusion_matrix_sklearn(gbrs, X_val, y_val)
No description has been provided for this image
# The RandomSearchCV tuned model is not overfitting the training data.
GridSearchCV

# prompt: libraries for gridsearchcv

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import recall_score, make_scorer, accuracy_score, precision_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
# defining model
model = AdaBoostClassifier(random_state=1)

# Parameter grid to pass in GridSearchCV
param_grid = {
    "n_estimators": np.arange(10, 110, 10),
    "learning_rate": [0.1, 0.01, 0.2, 0.05, 1],
    "estimator": [
        DecisionTreeClassifier(max_depth=1, random_state=1),
        DecisionTreeClassifier(max_depth=2, random_state=1),
        DecisionTreeClassifier(max_depth=3, random_state=1),
    ],
}
# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.recall_score)
# Create an AdaBoostClassifier instance
adaboost = AdaBoostClassifier(random_state=1)

# Create the GridSearchCV instance
grid_cv = GridSearchCV(estimator=adaboost, param_grid=param_grid, cv=5)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
# Fitting parameters in GridSeachCV
grid_cv.fit(X_train, y_train)
  GridSearchCV?i
best_estimator_: AdaBoostClassifier
estimator: DecisionTreeClassifier

 DecisionTreeClassifier?
print(
    "Best Parameters:{} \nScore: {}".format(grid_cv.best_params_, grid_cv.best_score_)
)
Best Parameters:{'estimator': DecisionTreeClassifier(max_depth=3, random_state=1), 'learning_rate': 0.2, 'n_estimators': 100} 
Score: 0.9636213991769547
# get best model
adb_tuned1 = grid_cv.best_estimator_
# fit the model
adb_tuned1.fit(X_train, y_train)
  AdaBoostClassifier?i
estimator: DecisionTreeClassifier

 DecisionTreeClassifier?
# Calculating different metrics on train set
Adaboost_grid_train = model_performance_classification_sklearn(
    adb_tuned1, X_train, y_train
)
print("Training performance:")
Adaboost_grid_train
Training performance:
Accuracy	Recall	Precision	F1
0	0.994239	0.977459	0.986556	0.981987
# creating confusion matrix
confusion_matrix_sklearn(adb_tuned1, X_train, y_train)
No description has been provided for this image
# Calculating different metrics on validation set
Adaboost_grid_val = model_performance_classification_sklearn(adb_tuned1, X_val, y_val)
print("Validation performance:")
Adaboost_grid_val
Validation performance:
Accuracy	Recall	Precision	F1
0	0.972853	0.886503	0.941368	0.913112
# creating confusion matrix
confusion_matrix_sklearn(adb_tuned1, X_val, y_val)
No description has been provided for this image
The tuned Adaboost model is not overfitting the training data. The model is good at identifying potential customers who would stay a customer and gives little False Negatives.
RandomizedSearchCV

from sklearn.ensemble import AdaBoostClassifier # Make sure AdaBoostClassifier is imported
model = AdaBoostClassifier(random_state=1) # defining model
# Parameter grid to pass in GridSearchCV
param_grid = {
    "n_estimators": np.arange(10, 110, 10),
    "learning_rate": [0.1, 0.01, 0.2, 0.05, 1],
    "estimator": [
        DecisionTreeClassifier(max_depth=1, random_state=1),
        DecisionTreeClassifier(max_depth=2, random_state=1),
        DecisionTreeClassifier(max_depth=3, random_state=1),
    ],
}
#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_jobs = -1, n_iter=50, scoring=scorer, cv=3, random_state=1)
#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train,y_train)
  RandomizedSearchCV?i
best_estimator_: AdaBoostClassifier
estimator: DecisionTreeClassifier

 DecisionTreeClassifier?
print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))
Best parameters are {'n_estimators': 90, 'learning_rate': 1, 'estimator': DecisionTreeClassifier(max_depth=2, random_state=1)} with CV score=0.8534560327198365:
# building model with best parameters
adb_tuned2 = randomized_cv.best_estimator_
# Fit the model on training data
adb_tuned2.fit(X_train, y_train)
  AdaBoostClassifier?i
estimator: DecisionTreeClassifier

 DecisionTreeClassifier?
# Calculating different metrics on train set
Adaboost_random_train = model_performance_classification_sklearn(
    adb_tuned2, X_train, y_train
)
print("Training performance:")
Adaboost_random_train
Training performance:
Accuracy	Recall	Precision	F1
0	0.996049	0.987705	0.987705	0.987705
# creating confusion matrix
confusion_matrix_sklearn(adb_tuned2, X_train, y_train)
No description has been provided for this image
# Calculating different metrics on validation set
Adaboost_random_val = model_performance_classification_sklearn(adb_tuned2, X_val, y_val)
print("Validation performance:")
Adaboost_random_val
Validation performance:
Accuracy	Recall	Precision	F1
0	0.968904	0.877301	0.925566	0.900787
# creating confusion matrix
confusion_matrix_sklearn(adb_tuned2, X_val, y_val)
No description has been provided for this image
The model is not overfitting.
The model is producing more False Negatives than previous models.
XGBoost

GridSearchCV

#defining model
model = XGBClassifier(random_state=1,eval_metric='logloss')
#Parameter grid to pass in GridSearchCV
param_grid={'n_estimators':np.arange(50,150,50),
            'scale_pos_weight':[2,5,10],
            'learning_rate':[0.01,0.1,0.2,0.05],
            'gamma':[0,1,3,5],
            'subsample':[0.8,0.9,1],
            'max_depth':np.arange(1,5,1),
            'reg_lambda':[5,10]}
# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.recall_score)
#Calling GridSearchCV
grid_cv = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scorer, cv=3, n_jobs = -1, verbose= 2)
#Fitting parameters in GridSeachCV
grid_cv.fit(X_train,y_train)
Fitting 3 folds for each of 2304 candidates, totalling 6912 fits
  GridSearchCV?i
best_estimator_: XGBClassifier

XGBClassifier
print("Best parameters are {} with CV score={}:" .format(grid_cv.best_params_,grid_cv.best_score_))
Best parameters are {'gamma': 0, 'learning_rate': 0.01, 'max_depth': 1, 'n_estimators': 50, 'reg_lambda': 5, 'scale_pos_weight': 10, 'subsample': 0.8} with CV score=1.0:
# building model with best parameters
xgb_tuned1 = grid_cv.best_estimator_

# Fit the model on training data
xgb_tuned1.fit(X_train, y_train)

 XGBClassifieri
XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric='logloss',
              feature_types=None, gamma=0, grow_policy=None,
              importance_type=None, interaction_constraints=None,
              learning_rate=0.01, max_bin=None, max_cat_threshold=None,
              max_cat_to_onehot=None, max_delta_step=None, max_depth=1,
              max_leaves=None, min_child_weight=None, missing=nan,
              monotone_constraints=None, multi_strategy=None, n_estimators=50,
              n_jobs=None, num_parallel_tree=None, random_state=1, ...)
# Calculating different metrics on train set
xgboost_grid_train = model_performance_classification_sklearn(
    xgb_tuned1, X_train, y_train
)
print("Training performance:")
xgboost_grid_train
Training performance:
Accuracy	Recall	Precision	F1
0	0.160658	1.0	0.160658	0.27684
# Calculating different metrics on validation set
xgboost_grid_val = model_performance_classification_sklearn(xgb_tuned1, X_val, y_val)
print("Validation performance:")
xgboost_grid_val
Validation performance:
Accuracy	Recall	Precision	F1
0	0.160908	1.0	0.160908	0.277211
The model has not overfitted the data.

RandomizedSearchCV

# defining model
model = XGBClassifier(random_state=1,eval_metric='logloss')
# Parameter grid to pass in RandomizedSearchCV
param_grid={'n_estimators':np.arange(50,150,50),
            'scale_pos_weight':[2,5,10],
            'learning_rate':[0.01,0.1,0.2,0.05],
            'gamma':[0,1,3,5],
            'subsample':[0.8,0.9,1],
            'max_depth':np.arange(1,5,1),
            'reg_lambda':[5,10]}
# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.recall_score)
#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=50, scoring=scorer, cv=3, random_state=1, n_jobs = -1)
#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train,y_train)
  RandomizedSearchCV?i
best_estimator_: XGBClassifier

XGBClassifier
print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))
Best parameters are {'subsample': 0.9, 'scale_pos_weight': 10, 'reg_lambda': 5, 'n_estimators': 50, 'max_depth': 1, 'learning_rate': 0.01, 'gamma': 1} with CV score=1.0:
# building model with best parameters
xgb_tuned2 = randomized_cv.best_estimator_

# Fit the model on training data
xgb_tuned2.fit(X_train, y_train)

 XGBClassifieri
XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric='logloss',
              feature_types=None, gamma=1, grow_policy=None,
              importance_type=None, interaction_constraints=None,
              learning_rate=0.01, max_bin=None, max_cat_threshold=None,
              max_cat_to_onehot=None, max_delta_step=None, max_depth=1,
              max_leaves=None, min_child_weight=None, missing=nan,
              monotone_constraints=None, multi_strategy=None, n_estimators=50,
              n_jobs=None, num_parallel_tree=None, random_state=1, ...)
# Calculating different metrics on train set
xgboost_random_train = model_performance_classification_sklearn(
    xgb_tuned2, X_train, y_train
)
print("Training performance:")
xgboost_random_train
Training performance:
Accuracy	Recall	Precision	F1
0	0.160658	1.0	0.160658	0.27684
# creating confusion matrix
confusion_matrix_sklearn(xgb_tuned2, X_train, y_train)
No description has been provided for this image
# Calculating different metrics on validation set
xgboost_random_val = model_performance_classification_sklearn(xgb_tuned2, X_val, y_val)
print("Validation performance:")
xgboost_random_val
Validation performance:
Accuracy	Recall	Precision	F1
0	0.160908	1.0	0.160908	0.277211
# creating confusion matrix
confusion_matrix_sklearn(xgb_tuned2, X_val, y_val)
No description has been provided for this image
Not overfitting the training and validation sets.

Oversampling with Tuned Models

Training set

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# print counts of target variable before oversampling
print("Before UpSampling, counts of label '1': {}".format(sum(y_train == 1)))
print("Before UpSampling, counts of label '0': {} \n".format(sum(y_train == 0)))

# Synthetic Minority Over Sampling Technique
sm = SMOTE(sampling_strategy=1, k_neighbors=5, random_state=1)

# fit and resample the training data
X_train_over, y_train_over = sm.fit_resample(X_train, y_train.ravel())

# prit data size after the oversampling
print("After UpSampling, counts of label '1': {}".format(sum(y_train_over == 1)))
print("After UpSampling, counts of label '0': {} \n".format(sum(y_train_over == 0)))


print("After UpSampling, the shape of train_X: {}".format(X_train_over.shape))
print("After UpSampling, the shape of train_y: {} \n".format(y_train_over.shape))
Before UpSampling, counts of label '1': 976
Before UpSampling, counts of label '0': 5099 

After UpSampling, counts of label '1': 5099
After UpSampling, counts of label '0': 5099 

After UpSampling, the shape of train_X: (10198, 29)
After UpSampling, the shape of train_y: (10198,) 

Validation set

print("Before UpSampling, counts of label '1': {}".format(sum(y_val == 1)))
print("Before UpSampling, counts of label '0': {} \n".format(sum(y_val == 0)))

sm = SMOTE(
    sampling_strategy=1, k_neighbors=5, random_state=1
)  # Synthetic Minority Over Sampling Technique
X_val_over, y_val_over = sm.fit_resample(X_val, y_val.ravel())


print("After UpSampling, counts of label '1': {}".format(sum(y_val_over == 1)))
print("After UpSampling, counts of label '0': {} \n".format(sum(y_val_over == 0)))


print("After UpSampling, the shape of train_X: {}".format(X_val_over.shape))
print("After UpSampling, the shape of train_y: {} \n".format(y_val_over.shape))
Before UpSampling, counts of label '1': 326
Before UpSampling, counts of label '0': 1700 

After UpSampling, counts of label '1': 1700
After UpSampling, counts of label '0': 1700 

After UpSampling, the shape of train_X: (3400, 29)
After UpSampling, the shape of train_y: (3400,) 

GradientBoost with oversampled

# fit the model
model = GradientBoostingClassifier(random_state=1)
gbost = model.fit(X_train_over, y_train_over)
# Calculating different metrics on train set
gradient_over = model_performance_classification_sklearn(
    gbost, X_train_over, y_train_over
)
print("Training performance:")
gradient_over
Training performance:
Accuracy	Recall	Precision	F1
0	0.980879	0.980584	0.981162	0.980873
# creating confusion matrix
confusion_matrix_sklearn(gbost, X_train_over, y_train_over)
No description has been provided for this image
# fit the model
model = GradientBoostingClassifier(random_state=1)
gbosv = model.fit(X_val_over, y_val_over)
# Calculating different metrics on train set
gradient_over_val = model_performance_classification_sklearn(
    gbosv, X_val_over, y_val_over
)
print("Validation performance:")
gradient_over_val
Validation performance:
Accuracy	Recall	Precision	F1
0	0.992353	0.993529	0.991197	0.992362
# creating confusion matrix
confusion_matrix_sklearn(gbosv, X_val_over, y_val_over)
No description has been provided for this image
AdaBoost with oversampled

# fit the model
model = AdaBoostClassifier(random_state=1)
adost = model.fit(X_train_over, y_train_over)
# Calculating different metrics on train set
Adaboost_over = model_performance_classification_sklearn(
    adost, X_train_over, y_train_over
)
print("Training performance:")
Adaboost_over
Training performance:
Accuracy	Recall	Precision	F1
0	0.967837	0.969014	0.966738	0.967875
# creating confusion matrix
confusion_matrix_sklearn(adost, X_val_over, y_val_over)
No description has been provided for this image
# fit the model
model = AdaBoostClassifier(random_state=1)
adosv = model.fit(X_val_over, y_val_over)
# Calculating different metrics on train set
Adaboost_over_val = model_performance_classification_sklearn(
    adosv, X_val_over, y_val_over
)
print("Validation performance:")
Adaboost_over_val
Validation performance:
Accuracy	Recall	Precision	F1
0	0.973529	0.975882	0.971311	0.973592
# creating confusion matrix
confusion_matrix_sklearn(adosv, X_val_over, y_val_over)
No description has been provided for this image
XGBoost with oversampled

# defining model
model = XGBClassifier(random_state=1, eval_metric="logloss")
xgbost = model.fit(X_train_over, y_train_over)
# Calculating different metrics on train set
xgboost_over = model_performance_classification_sklearn(
    xgbost, X_train_over, y_train_over
)
print("Training performance:")
xgboost_over
Training performance:
Accuracy	Recall	Precision	F1
0	1.0	1.0	1.0	1.0
# creating confusion matrix
confusion_matrix_sklearn(xgbost, X_train_over, y_train_over)
No description has been provided for this image
# fit the model
model = XGBClassifier(random_state=1, eval_metric="logloss")
xgbosv = model.fit(X_val_over, y_val_over)
# Calculating different metrics on train set
xgboost_over_val = model_performance_classification_sklearn(
    xgbosv, X_val_over, y_val_over
)
print("Validation performance:")
xgboost_over_val
Validation performance:
Accuracy	Recall	Precision	F1
0	1.0	1.0	1.0	1.0
# creating confusion matrix
confusion_matrix_sklearn(xgbosv, X_val_over, y_val_over)
No description has been provided for this image
UnderSampling with Tuned Models

Training set

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# define undersample strategy
undersample = RandomUnderSampler(sampling_strategy=0.5)

# fit and apply the transform
X_under, y_under = undersample.fit_resample(X_train, y_train)
display(X_train.shape)
display(X_under.shape)
(6075, 29)
(2928, 29)
Validation set

# define undersample strategy
undersample = RandomUnderSampler(sampling_strategy=0.5)

# fit and apply the transform
X_under_val, y_under_val = undersample.fit_resample(X_val, y_val)
display(X_val.shape)
display(X_under_val.shape)
(2026, 29)
(978, 29)
GradientBoost with undersampled

# fit the model
model = GradientBoostingClassifier(random_state=1)
gbust = model.fit(X_under, y_under)
# Calculating different metrics on train set
gradient_under = model_performance_classification_sklearn(gbust, X_under, y_under)
print("Training performance:")
gradient_under
Training performance:
Accuracy	Recall	Precision	F1
0	0.97097	0.951844	0.960703	0.956253
# creating confusion matrix
confusion_matrix_sklearn(gbust, X_under, y_under)
No description has been provided for this image
# fit the model
model = GradientBoostingClassifier(random_state=1)
gbusv = model.fit(X_under_val, y_under_val)
# Calculating different metrics on train set
gradient_under_val = model_performance_classification_sklearn(
    gbusv, X_under_val, y_under_val
)
print("Validation performance:")
gradient_under_val
Validation performance:
Accuracy	Recall	Precision	F1
0	0.992843	0.993865	0.984802	0.989313
# creating confusion matrix
confusion_matrix_sklearn(gbusv, X_under_val, y_under_val)
No description has been provided for this image
AdaBoost with undersampled

# fit the model
model = AdaBoostClassifier(random_state=1)
adust = model.fit(X_under, y_under)
# Calculating different metrics on train set
Adaboost_under = model_performance_classification_sklearn(adust, X_under, y_under)
print("Training performance:")
Adaboost_under
Training performance:
Accuracy	Recall	Precision	F1
0	0.950137	0.919057	0.930498	0.924742
# creating confusion matrix
confusion_matrix_sklearn(adust, X_under, y_under)
No description has been provided for this image
# fit the model
model = AdaBoostClassifier(random_state=1)
adusv = model.fit(X_under_val, y_under_val)
# Calculating different metrics on train set
Adaboost_under_val = model_performance_classification_sklearn(
    adusv, X_under_val, y_under_val
)
print("Validation performance:")
Adaboost_under_val
Validation performance:
Accuracy	Recall	Precision	F1
0	0.96728	0.95092	0.95092	0.95092
# creating confusion matrix
confusion_matrix_sklearn(adusv, X_under_val, y_under_val)
No description has been provided for this image
XGBoost with undersampled

# defining model
model = XGBClassifier(random_state=1, eval_metric="logloss")
xgbust = model.fit(X_under, y_under)
display(xgbust)

 XGBClassifieri
XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric='logloss',
              feature_types=None, gamma=None, grow_policy=None,
              importance_type=None, interaction_constraints=None,
              learning_rate=None, max_bin=None, max_cat_threshold=None,
              max_cat_to_onehot=None, max_delta_step=None, max_depth=None,
              max_leaves=None, min_child_weight=None, missing=nan,
              monotone_constraints=None, multi_strategy=None, n_estimators=None,
              n_jobs=None, num_parallel_tree=None, random_state=1, ...)
# Calculating different metrics on train set
xgboost_under = model_performance_classification_sklearn(xgbust, X_under, y_under)
print("Training performance:")
xgboost_under
Training performance:
Accuracy	Recall	Precision	F1
0	1.0	1.0	1.0	1.0
# creating confusion matrix
confusion_matrix_sklearn(xgbust, X_under, y_under)
No description has been provided for this image
# defining model
model = XGBClassifier(random_state=1, eval_metric="logloss")
xgbusv = model.fit(X_under_val, y_under_val)
# Calculating different metrics on train set
xgboost_under_val = model_performance_classification_sklearn(
    xgbusv, X_under_val, y_under_val
)
print("Validation performance:")
xgboost_under_val
Validation performance:
Accuracy	Recall	Precision	F1
0	1.0	1.0	1.0	1.0
# creating confusion matrix
confusion_matrix_sklearn(xgbusv, X_under_val, y_under_val)
No description has been provided for this image
Model Selection

# concatenate the dataframes
models_train_comp_df = pd.concat(
    [
        gradient_grid_train.T,
        gradient_grid_val.T,
        gradient_random_train.T,
        gradient_random_val.T,
        gradient_over.T,
        gradient_over_val.T,
        gradient_under.T,
        gradient_under_val.T,
        Adaboost_grid_train.T,
        Adaboost_grid_val.T,
        Adaboost_random_train.T,
        Adaboost_random_val.T,
        Adaboost_over.T,
        Adaboost_over_val.T,
        Adaboost_under.T,
        Adaboost_under_val.T,
        xgboost_grid_train.T,
        xgboost_grid_val.T,
        xgboost_random_train.T,
        xgboost_random_val.T,
        xgboost_over.T,
        xgboost_over_val.T,
        xgboost_under.T,
        xgboost_under_val.T,
    ],
    axis=1,
)
# give the columns headers
models_train_comp_df.columns = [
    "GradientBoost Tuned with Grid search Training Set",
    "GradientBoost Tuned with Grid search Validation Set",
    "GradientBoost Tuned with Random search Training Set",
    "GradientBoost Tuned with Random search Validation Set",
    "GradientBoost Tuned with Oversampled Training Set",
    "GradientBoost Tuned with Oversampled Validation Set",
    "GradientBoost Tuned with Undersampled Training Set",
    "GradientBoost Tuned with Undersampled Validation Set",
    "AdaBoost Tuned with Grid search Training Set",
    "AdaBoost Tuned with Grid search Validation Set",
    "AdaBoost Tuned with Random search Training Set",
    "AdaBoost Tuned with Random search Validation Set",
    "AdaBoost Tuned with Oversampled Training Set",
    "AdaBoost Tuned with Oversampled Validation Set",
    "AdaBoost Tuned with Undersampled Training Set",
    "AdaBoost Tuned with Undersampled Validation Set",
    "Xgboost Tuned with Grid search Training Set",
    "Xgboost Tuned with Grid search Validation Set",
    "Xgboost Tuned with Random Search Training Set",
    "Xgboost Tuned with Random Search Validation Set",
    "Xgboost Tuned with Oversampled Training Set",
    "Xgboost Tuned with Oversampled Validation Set",
    "Xgboost Tuned with Undersampled Training Set",
    "Xgboost Tuned with Undersampled Validation Set",
]
# print the dataframe
print("Training & Validation performance comparison:")
models_train_comp_df
Training & Validation performance comparison:
GradientBoost Tuned with Grid search Training Set	GradientBoost Tuned with Grid search Validation Set	GradientBoost Tuned with Random search Training Set	GradientBoost Tuned with Random search Validation Set	GradientBoost Tuned with Oversampled Training Set	GradientBoost Tuned with Oversampled Validation Set	GradientBoost Tuned with Undersampled Training Set	GradientBoost Tuned with Undersampled Validation Set	AdaBoost Tuned with Grid search Training Set	AdaBoost Tuned with Grid search Validation Set	...	AdaBoost Tuned with Undersampled Training Set	AdaBoost Tuned with Undersampled Validation Set	Xgboost Tuned with Grid search Training Set	Xgboost Tuned with Grid search Validation Set	Xgboost Tuned with Random Search Training Set	Xgboost Tuned with Random Search Validation Set	Xgboost Tuned with Oversampled Training Set	Xgboost Tuned with Oversampled Validation Set	Xgboost Tuned with Undersampled Training Set	Xgboost Tuned with Undersampled Validation Set
Accuracy	0.986337	0.973346	0.981893	0.972359	0.980879	0.992353	0.970970	0.992843	0.994239	0.972853	...	0.950137	0.96728	0.160658	0.160908	0.160658	0.160908	1.0	1.0	1.0	1.0
Recall	0.940574	0.895706	0.929303	0.883436	0.980584	0.993529	0.951844	0.993865	0.977459	0.886503	...	0.919057	0.95092	1.000000	1.000000	1.000000	1.000000	1.0	1.0	1.0	1.0
Precision	0.973489	0.935897	0.956751	0.941176	0.981162	0.991197	0.960703	0.984802	0.986556	0.941368	...	0.930498	0.95092	0.160658	0.160908	0.160658	0.160908	1.0	1.0	1.0	1.0
F1	0.956748	0.915361	0.942827	0.911392	0.980873	0.992362	0.956253	0.989313	0.981987	0.913112	...	0.924742	0.95092	0.276840	0.277211	0.276840	0.277211	1.0	1.0	1.0	1.0
4 rows × 24 columns

# create a dataframe from the confusion matrix false negative scores
data = {
    "Gradient Boost Tuned with Gridsearch Training Set": 58,
    "Gradient Boost Tuned with Gridsearch Validation Set": 34,
    "Gradient Boost Tuned with Randomsearch Training Set": 69,
    "Gradient Boost Tuned with Randomsearch Validation Set": 38,
    "Gradient Boost Tuned with Oversampled Training Set": 99,
    "Gradient Boost Tuned with Oversampled Validation Set": 11,
    "Gradient Boost Tuned with Undersampled Training Set": 40,
    "Gradient Boost Tuned with Undersampled Validation Set": 2,
    "AdaBoost Tuned with Gridsearch Training Set": 49,
    "AdaBoost Tuned with Gridsearch Validation Set": 36,
    "AdaBoost Tuned with Randomsearch Training Set": 12,
    "AdaBoost Tuned with Randomsearch Validation Set": 40,
    "AdaBoost Tuned with Oversampled Training Set": 57,
    "AdaBoost Tuned with Oversampled Validation Set": 41,
    "AdaBoost Tuned with Undersampled Training Set": 70,
    "AdaBoost Tuned with Undersampled Validation Set": 16,
    "Xgboost Tuned with Gridsearch Training Set": 21,
    "Xgboost Tuned with Gridsearch Validation Set": 3,
    "Xgboost Tuned with RandomSearch Training Set": 13,
    "Xgboost Tuned with RandomSearch Validation Set": 12,
    "Xgboost Tuned with Oversampled Training Set": 0,
    "Xgboost Tuned with Oversampled Validation Set": 0,
    "Xgboost Tuned with Undersampled Training Set": 0,
    "Xgboost Tuned with Undersampled Validation Set": 0,
}
false_negatives_df_train = pd.DataFrame.from_dict(
    data, orient="index", columns=["False Negatives"]
)
display(false_negatives_df_train.sort_values("False Negatives"))
False Negatives
Xgboost Tuned with Oversampled Training Set	0
Xgboost Tuned with Oversampled Validation Set	0
Xgboost Tuned with Undersampled Training Set	0
Xgboost Tuned with Undersampled Validation Set	0
Gradient Boost Tuned with Undersampled Validation Set	2
Xgboost Tuned with Gridsearch Validation Set	3
Gradient Boost Tuned with Oversampled Validation Set	11
AdaBoost Tuned with Randomsearch Training Set	12
Xgboost Tuned with RandomSearch Validation Set	12
Xgboost Tuned with RandomSearch Training Set	13
AdaBoost Tuned with Undersampled Validation Set	16
Xgboost Tuned with Gridsearch Training Set	21
Gradient Boost Tuned with Gridsearch Validation Set	34
AdaBoost Tuned with Gridsearch Validation Set	36
Gradient Boost Tuned with Randomsearch Validation Set	38
Gradient Boost Tuned with Undersampled Training Set	40
AdaBoost Tuned with Randomsearch Validation Set	40
AdaBoost Tuned with Oversampled Validation Set	41
AdaBoost Tuned with Gridsearch Training Set	49
AdaBoost Tuned with Oversampled Training Set	57
Gradient Boost Tuned with Gridsearch Training Set	58
Gradient Boost Tuned with Randomsearch Training Set	69
AdaBoost Tuned with Undersampled Training Set	70
Gradient Boost Tuned with Oversampled Training Set	99
Model Comparison Observations

Overfitting: Most of the models are not overfitting.

Recall Score: Are generally above 85%.

False Negatives: Are very low for all models.

Performance on the test set

Recall & Precision > 0.95 and find test accuracy > 0.70 on test set.

# Calculating different metrics on the test set
gbsov_test = model_performance_classification_sklearn(gbosv, X_test, y_test)
print("GradientBoost Tuned with Oversampled Validation Set: Test performance:")
gbsov_test
GradientBoost Tuned with Oversampled Validation Set: Test performance:
Accuracy	Recall	Precision	F1
0	0.953603	0.873846	0.84273	0.858006
# Calculating different metrics on the test set
adaboost_test = model_performance_classification_sklearn(adb_tuned2, X_test, y_test)
print("AdaBoost Tuned with Random search Training Set: Test performance:")
adaboost_test
AdaBoost Tuned with Random search Training Set: Test performance:
Accuracy	Recall	Precision	F1
0	0.975814	0.929231	0.920732	0.924962
# Calculating different metrics on the test set
gbusv_test = model_performance_classification_sklearn(gbusv, X_test, y_test)
print("GradientBoost Tuned with Undersampled Validation Set: Test performance:")
gbusv_test
GradientBoost Tuned with Undersampled Validation Set: Test performance:
Accuracy	Recall	Precision	F1
0	0.950148	0.901538	0.809392	0.852984
None of these models performed above 95% on Recall and Precision and above 70% in accuracy

# Calculating different metrics on the test set
xgbust_test = model_performance_classification_sklearn(xgbust, X_test, y_test)
print("XGBoost with UnderSampled Training Data: Test performance:")
xgbust_test
XGBoost with UnderSampled Training Data: Test performance:
Accuracy	Recall	Precision	F1
0	0.966436	0.953846	0.853994	0.901163
# creating confusion matrix
confusion_matrix_sklearn(xgbust, X_test, y_test)
No description has been provided for this image
This model is giving us the accuracy, recall and precision we need, along with the low false negative error rate

# get list of features from dmodel
feature_names = X.columns
importances = xgbust.feature_importances_
indices = np.argsort(importances)

# plot the features
plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()
No description has been provided for this image
# create a scatterplot
sns.scatterplot(x=df.Total_Trans_Ct, y=df.Attrition_Flag)
plt.show()
No description has been provided for this image
# create a scatterplot
sns.scatterplot(x=df.Total_Revolving_Bal, y=df.Attrition_Flag)
<Axes: xlabel='Total_Revolving_Bal', ylabel='Attrition_Flag'>
No description has been provided for this image
# crosstabs
pd.crosstab(index=df["Attrition_Flag"], columns=df["Total_Relationship_Count"])
Total_Relationship_Count	1	2	3	4	5	6
Attrition_Flag						
Attrited Customer	233	346	400	225	227	196
Existing Customer	677	897	1905	1687	1664	1670
# create a scatterplot
sns.scatterplot(
    x=df.Total_Relationship_Count,
    y=df.Total_Relationship_Count,
    hue=df["Attrition_Flag"],
)
<Axes: xlabel='Total_Relationship_Count', ylabel='Total_Relationship_Count'>
No description has been provided for this image
Pipelines for productionizing the model

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
test_data = df.copy()
type(data)
test_data.head()
CLIENTNUM	Attrition_Flag	Customer_Age	Gender	Dependent_count	Education_Level	Marital_Status	Income_Category	Card_Category	Months_on_book	...	Months_Inactive_12_mon	Contacts_Count_12_mon	Credit_Limit	Total_Revolving_Bal	Avg_Open_To_Buy	Total_Amt_Chng_Q4_Q1	Total_Trans_Amt	Total_Trans_Ct	Total_Ct_Chng_Q4_Q1	Avg_Utilization_Ratio
0	768805383	Existing Customer	45	M	3	High School	Married	
60
K
−
80K	Blue	39	...	1	3	12691.0	777	11914.0	1.335	1144	42	1.625	0.061
1	818770008	Existing Customer	49	F	5	Graduate	Single	Less than $40K	Blue	44	...	1	2	8256.0	864	7392.0	1.541	1291	33	3.714	0.105
2	713982108	Existing Customer	51	M	3	Graduate	Married	
80
K
−
120K	Blue	36	...	1	0	3418.0	0	3418.0	2.594	1887	20	2.333	0.000
3	769911858	Existing Customer	40	F	4	High School	NaN	Less than $40K	Blue	34	...	4	1	3313.0	2517	796.0	1.405	1171	20	2.333	0.760
4	709106358	Existing Customer	40	M	3	Uneducated	Married	
60
K
−
80K	Blue	21	...	1	0	4716.0	0	4716.0	2.175	816	28	2.500	0.000
5 rows × 21 columns

# Separating target variable and other variables
X = test_data.drop(columns=["Attrition_Flag"])
Y = test_data["Attrition_Flag"]
# check head
X.head()
CLIENTNUM	Customer_Age	Gender	Dependent_count	Education_Level	Marital_Status	Income_Category	Card_Category	Months_on_book	Total_Relationship_Count	Months_Inactive_12_mon	Contacts_Count_12_mon	Credit_Limit	Total_Revolving_Bal	Avg_Open_To_Buy	Total_Amt_Chng_Q4_Q1	Total_Trans_Amt	Total_Trans_Ct	Total_Ct_Chng_Q4_Q1	Avg_Utilization_Ratio
0	768805383	45	M	3	High School	Married	
60
K
−
80K	Blue	39	5	1	3	12691.0	777	11914.0	1.335	1144	42	1.625	0.061
1	818770008	49	F	5	Graduate	Single	Less than $40K	Blue	44	6	1	2	8256.0	864	7392.0	1.541	1291	33	3.714	0.105
2	713982108	51	M	3	Graduate	Married	
80
K
−
120K	Blue	36	4	1	0	3418.0	0	3418.0	2.594	1887	20	2.333	0.000
3	769911858	40	F	4	High School	NaN	Less than $40K	Blue	34	3	4	1	3313.0	2517	796.0	1.405	1171	20	2.333	0.760
4	709106358	40	M	3	Uneducated	Married	
60
K
−
80K	Blue	21	5	1	0	4716.0	0	4716.0	2.175	816	28	2.500	0.000
# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.30, random_state=1, stratify=Y
)
print(X_train.shape, X_test.shape)
display(X_train.head())
display(y_train)
(7088, 20) (3039, 20)
CLIENTNUM	Customer_Age	Gender	Dependent_count	Education_Level	Marital_Status	Income_Category	Card_Category	Months_on_book	Total_Relationship_Count	Months_Inactive_12_mon	Contacts_Count_12_mon	Credit_Limit	Total_Revolving_Bal	Avg_Open_To_Buy	Total_Amt_Chng_Q4_Q1	Total_Trans_Amt	Total_Trans_Ct	Total_Ct_Chng_Q4_Q1	Avg_Utilization_Ratio
678	710280558	51	M	1	Graduate	Single	
60
K
−
80K	Blue	39	3	3	2	8796.0	2517	6279.0	0.492	1195	18	0.800	0.286
7524	713920158	41	F	3	Uneducated	NaN	abc	Blue	36	2	3	1	13733.0	0	13733.0	0.325	1591	30	0.875	0.000
8725	824177733	53	F	2	High School	Single	Less than $40K	Blue	49	1	2	3	9678.0	1710	7968.0	0.745	7682	90	0.579	0.177
10029	715268508	42	M	3	Graduate	Married	
40
K
−
60K	Gold	36	3	2	5	23981.0	1399	22582.0	0.712	14840	125	0.761	0.058
1383	710872233	27	M	0	NaN	Single	
40
K
−
60K	Blue	17	5	1	2	4610.0	0	4610.0	0.794	2280	49	0.400	0.000
Attrition_Flag
678	Existing Customer
7524	Attrited Customer
8725	Existing Customer
10029	Existing Customer
1383	Existing Customer
...	...
509	Attrited Customer
8365	Existing Customer
7169	Existing Customer
6113	Attrited Customer
10034	Attrited Customer
7088 rows × 1 columns


dtype: object
# Transform the target variable from string to int.
target_variable = "Attrition_Flag"

y_train.replace(
    to_replace={"Attrited Customer": 1, "Existing Customer": 0}
)


# to drop unnecessary columns
drop_cols = "CLIENTNUM"
drop_transformer = X_train.drop(columns=drop_cols, inplace=True)
# creating a list of numerical variables
numerical_features = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
]
# creating a transformer for numerical variables, which will apply simple imputer on the numerical variables
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
# creating a list of categorical variables
categorical_features = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
# creating a transformer for categorical variables, which will first apply simple imputer and
# then do one hot encoding for categorical variables
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),  # handle missing
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[  # List of (name, transformer, columns)
        ("num_step", numeric_transformer, numerical_features),
        ("cat_step", categorical_transformer, categorical_features),
    ],
    remainder="passthrough",
    n_jobs=-1,
    verbose=True,
)
# Creating new pipeline with best parameters
production_model = Pipeline(
    steps=[
        ("pre", preprocessor),  # pipelines from above
        (
            "XGB",  # best model for prediction
            XGBClassifier(
                base_score=0.5,
                booster="gbtree",
                colsample_bylevel=1,
                colsample_bynode=1,
                colsample_bytree=1,
                eval_metric="logloss",
                gamma=0,
                gpu_id=-1,
                importance_type="gain",
                interaction_constraints="",
                learning_rate=0.300000012,
                max_delta_step=0,
                max_depth=6,
                min_child_weight=1,
                missing=np.nan,
                monotone_constraints="()",
                n_estimators=100,
                n_jobs=4,
                num_parallel_tree=1,
                random_state=1,
                reg_alpha=0,
                reg_lambda=1,
                scale_pos_weight=1,
                subsample=1,
                tree_method="exact",
                validate_parameters=1,
                verbosity=None,
            ),
        ),
    ]
)
# view pipeline
display(production_model[0])
display(type(production_model))
  ColumnTransformer?i
num_step

 SimpleImputer?
cat_step

 SimpleImputer?

 OneHotEncoder?
remainder

passthrough
sklearn.pipeline.Pipeline
def __init__(steps, *, memory=None, verbose=False)
A sequence of data transformers with an optional final predictor.

`Pipeline` allows you to sequentially apply a list of transformers to
preprocess the data and, if desired, conclude the sequence with a final
:term:`predictor` for predictive modeling.

Intermediate steps of the pipeline must be 'transforms', that is, they
must implement `fit` and `transform` methods.
The final :term:`estimator` only needs to implement `fit`.
The transformers in the pipeline can be cached using ``memory`` argument.

The purpose of the pipeline is to assemble several steps that can be
cross-validated together while setting different parameters. For this, it
enables setting parameters of the various steps using their names and the
parameter name separated by a `'__'`, as in the example below. A step's
estimator may be replaced entirely by setting the parameter with its name
to another estimator, or a transformer removed by setting it to
`'passthrough'` or `None`.

For an example use case of `Pipeline` combined with
:class:`~sklearn.model_selection.GridSearchCV`, refer to
:ref:`sphx_glr_auto_examples_compose_plot_compare_reduction.py`. The
example :ref:`sphx_glr_auto_examples_compose_plot_digits_pipe.py` shows how
to grid search on a pipeline using `'__'` as a separator in the parameter names.

Read more in the :ref:`User Guide <pipeline>`.

.. versionadded:: 0.5

Parameters
----------
steps : list of tuples
    List of (name of step, estimator) tuples that are to be chained in
    sequential order. To be compatible with the scikit-learn API, all steps
    must define `fit`. All non-last steps must also define `transform`. See
    :ref:`Combining Estimators <combining_estimators>` for more details.

memory : str or object with the joblib.Memory interface, default=None
    Used to cache the fitted transformers of the pipeline. The last step
    will never be cached, even if it is a transformer. By default, no
    caching is performed. If a string is given, it is the path to the
    caching directory. Enabling caching triggers a clone of the transformers
    before fitting. Therefore, the transformer instance given to the
    pipeline cannot be inspected directly. Use the attribute ``named_steps``
    or ``steps`` to inspect estimators within the pipeline. Caching the
    transformers is advantageous when fitting is time consuming.

verbose : bool, default=False
    If True, the time elapsed while fitting each step will be printed as it
    is completed.

Attributes
----------
named_steps : :class:`~sklearn.utils.Bunch`
    Dictionary-like object, with the following attributes.
    Read-only attribute to access any step parameter by user given name.
    Keys are step names and values are steps parameters.

classes_ : ndarray of shape (n_classes,)
    The classes labels. Only exist if the last step of the pipeline is a
    classifier.

n_features_in_ : int
    Number of features seen during :term:`fit`. Only defined if the
    underlying first estimator in `steps` exposes such an attribute
    when fit.

    .. versionadded:: 0.24

feature_names_in_ : ndarray of shape (`n_features_in_`,)
    Names of features seen during :term:`fit`. Only defined if the
    underlying estimator exposes such an attribute when fit.

    .. versionadded:: 1.0

See Also
--------
make_pipeline : Convenience function for simplified pipeline construction.

Examples
--------
>>> from sklearn.svm import SVC
>>> from sklearn.preprocessing import StandardScaler
>>> from sklearn.datasets import make_classification
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.pipeline import Pipeline
>>> X, y = make_classification(random_state=0)
>>> X_train, X_test, y_train, y_test = train_test_split(X, y,
...                                                     random_state=0)
>>> pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
>>> # The pipeline can be used as any other estimator
>>> # and avoids leaking the test set into the train set
>>> pipe.fit(X_train, y_train).score(X_test, y_test)
0.88
>>> # An estimator's parameter can be set using '__' syntax
>>> pipe.set_params(svc__C=10).fit(X_train, y_train).score(X_test, y_test)
0.76
# prompt: y unique values
unique_values = np.unique(y_val)
unique_values
array([0, 1])
# Check unique values in y_train
print(np.unique(y_train))
['Attrited Customer' 'Existing Customer']
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
print(np.unique(y_train_encoded))
[0 1]
# prompt: fit the production pipeline  on training data

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler


production_model.fit(X_train, y_train_encoded)
  Pipeline?i
 pre: ColumnTransformer?
num_step

 SimpleImputer?
cat_step

 SimpleImputer?

 OneHotEncoder?
remainder

passthrough

XGBClassifier
Business recomendations
Total Transaction Count: Customers with fewer transactions are more likely to attrite.

Month-to-Month Balance: A low balance carry-over suggests less engagement and a higher likelihood of attrition.

Product Holdings: Customers with fewer products (between 2 and 3) tend to show signs of potential attrition.

Customers with total transactions below 100 should be flagged for further analysis and engagement.

Customers with a monthly balance carry-over below 500 should be considered at risk and targeted for retention.

Customers holding 2-3 products should be monitored closely, as they may be on the cusp of attrition.

Actionable insights

For customers falling into these thresholds, the bank could provide personalized offers such as fee waivers, rewards for higher transaction volumes, special interest rates to encourage increased retention.

Provide cash back or rewards points for customers who surpass a certain number of transactions per month

Encourage customers to consolidate or open more accounts by offering a bonus for increasing product holdings

Develop credit cards with lower credit limits, eligibility requirements tailored for these customers.

These cards can come with lower annual fees and rewards geared towards small purchases or essential expenses.

Monitor how these products impact retention rates and customer satisfaction among these segments.

Introduce credit cards with lower income criteria and simplified benefits to cater to the needs of the most vulnerable segments.

Regularly monitor the thresholds for transaction volume, balance, and product holdings to ensure early identification of at-risk customers.

Use customer segmentation techniques to further refine retention efforts, focusing on the most vulnerable segments first.

Cross-sell or upsell additional products (e.g., credit cards, loans, insurance) to strengthen the customer-bank relationship.

Bundle services into packages with exclusive benefits for customers who adopt multiple products.

Identify gaps in product offerings for these customers and introduce tailored solutions that address their unmet needs.

Create age-specific marketing campaigns to cater to diverse banking needs.

Introduce product upgrades or lifecycle-based offerings to keep customers engaged as their financial needs evolve.

Offer financial counseling or credit optimization plans for high-utilization customers.

Increase credit limits for loyal customers to provide flexibility and incentivize continued use.

Establish proactive communication strategies, like regular account reviews, personalized offers, or updates about new services.

Enhance customer support channels (chat, email, phone) to ensure customers feel valued and supported

Implement targeted communications to educate customers on ways to optimize their savings and balances, such as auto-savings plans or financial planning tools.

Offer low-balance customers incentives, like higher interest rates on balances above a certain threshold.

Monitor and reach out to customers who consistently maintain low balances with personalized offers to deepen their financial involvement with the bank.

