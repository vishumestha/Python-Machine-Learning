import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use("fivethirtyeight")


we_stats={"Day":[1,2,3,4,5,6],"Visitors":[43,24,23,67,45,67],"Bonus_rate":[45,23,67,89,45,60]}
df=pd.DataFrame(we_stats)

print(df.head())
#Set the Day as index
#df.set_index("Day")
#print(df.head())

#Set the Day as index 
#df=df.set_index("Day")
#print(df.head())

#set the Day as index
df.set_index("Day",inplace=True)
print(df.head())   

#plot the visitors
df['Visitors'].plot()
plt.show()

print(['Visitors'])
print(df[['Visitors','Bonus_rate']])

#convert datafram to list
print(df['Visitors'].tolist())

#convert dataframe to 2-dimension list
print(np.array(df[['Visitors','Bonus_rate']]))
df.plot()
plt.show()
#------------------------------------------------------------------------------
#tutorial 3
#Read the CSV file
df3=pd.read_csv('D:/Data Science with Python/Pandas/ZILL-Z77006_3B.csv')
df3.head()
df3.set_index("Date",inplace=True)
print(df3.head())
#
#Copy the the  to csv file
#df3.to_csv("names.csv",header=False)
#print(df3.head())
#
#Copy the the specific coloums to csv file
#df3["Value"].to_csv("names21.csv",header=True)
#df3=pd.read_csv("D:/Data Science with Python/Pandas/names21.csv")
#df3.head()

#Change the name of the column, u have to specify all the colunm names
df3.columns=["House_Rent"]
df3.head()

#Chnage the name of the specific colums
df3.rename(columns={"House_Rent":"Price"},inplace=True)
df3.head()

#What if the file doesn't have headers? No problem
#df3 = pd.read_csv('newcsv4.csv', names = ['Date','House_Price'], index_col=0)
#print(df3.head())

#convert to the html file
#df.read_html("names.html")
#------------------------------------------------------------------------------





