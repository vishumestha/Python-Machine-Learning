import quandl
import html5lib
import pandas as pd

#To extract the data from quandl
#df=quandl.get("FMAC/HPI_AK", authtoken="SjWfQrbqTXLCNy7ngaXk")
#print(df)

#extract the data from the URL
fiddy_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')

print(fiddy_states)#Whole data
print(fiddy_states[0])# The relavent Table
print(fiddy_states[0][0])# Relavent Column
print(fiddy_states[0][0][1:])#Satrting from firts row

for abbr in fiddy_states[0][0][1:]:
    print("FMAC/HPI_"+str(abbr))
#-----------------------------------------------------------------------------------------
#Tutorial5

df1 = pd.DataFrame({'HPI':[80,85,88,85],
                    'Int_rate':[2, 3, 2, 2],
                    'US_GDP_Thousands':[50, 55, 65, 55]},
                   index = [2001, 2002, 2003, 2004])

df2 = pd.DataFrame({'HPI':[80,85,88,85],
                    'Int_rate':[2, 3, 2, 2],
                    'US_GDP_Thousands':[50, 55, 65, 55]},
                   index = [2005, 2006, 2007, 2008])

df3 = pd.DataFrame({'HPI':[80,85,88,85],
                    'Int_rate':[2, 3, 2, 2],
                    'Low_tier_HPI':[50, 52, 50, 53]},
                    index =[2001, 2002, 2003, 2004])

#It will  concatinate 3 data frames
concat=pd.concat([df1,df2,df3])
print(concat)

#append
df4=df1.append(df3,)
print(df4)

#s=pd.Series([91,3,55]) we need to specify index otherwise we cannot append to particular column
s=pd.Series([91,3,55],index=['HPI','Int_rate','US_GDP_Thousands'])
df5=df1.append(s,ignore_index=True)
print(df5)




