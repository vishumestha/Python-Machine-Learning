import pandas as pd

df1 = pd.DataFrame({'HPI':[80,85,88,85],
                    'Int_rate':[2, 3, 2, 2],
                    'US_GDP_Thousands':[50, 55, 65, 55]},
                   index = [2001, 2002, 2003, 2004])

df2 = pd.DataFrame({'HPI':[80,85,88,85],
                    'Int_rate':[2, 3, 2, 2],
                    'US_GDP_Thousands':[50, 55, 65, 55]},
                   index = [2005, 2006, 2007, 2008])

df3 = pd.DataFrame({'HPI':[80,85,88,85],
                    'Unemployment':[7, 8, 9, 6],
                    'Low_tier_HPI':[50, 52, 50, 53]},
                   index = [2001, 2002, 2003, 2004])


df4=pd.merge(df1,df2,on=["HPI","Int_rate"])
#print(df1)
#print(df2)
print(df4)

df4.set_index("HPI",inplace=True)
print(df4)
#Merge will join the two data frame based on the specified column the resultent data frame wont conatin the index
#join will join the table based on the index

#How to name the idex
df1.index.name="Year"
df2.index.name="Year"
df3.index.name="Year"
print(df1)
print(df2)
#df5=df1.join(df2)

#df1.set_index('HPI', inplace=True)
#df3.set_index('HPI', inplace=True)

#df5=df1.join(df2)# ValueError: columns overlap but no suffix specified:
df5=df1.join(df2,how='left', lsuffix='_left', rsuffix='_right')
print(df5)

#Left - equal to left outer join SQL - use keys from left frame only
#Right - right outer join from SQL- use keys from right frame only.
#Outer - full outer join - use union of keys
#Inner - use only intersection of keys.
df6=pd.merge(df1,df3,on="HPI",how="left")
print(df6)




