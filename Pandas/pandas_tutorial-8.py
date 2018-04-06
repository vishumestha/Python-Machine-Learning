import pandas as pd
import numpy as np
import quandl
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use("fivethirtyeight")

#Pickle, which will convert your object to a byte stream, or the reverse with unpickling

#data_Real_estate=quandl.get("FMAC/HPI_AK", authtoken="SjWfQrbqTXLCNy7ngaXk")
#print(data_Real_estate)

df=pd.read_html("https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States")
df=df[0][1][2:]
print(df)

main_data=pd.DataFrame()

for abbr in df:
    query="FMAC/HPI_"+str(abbr)
    data_Real_estate=quandl.get(query, authtoken="SjWfQrbqTXLCNy7ngaXk")
    data_Real_estate.columns=[abbr]#data_Real_estate.rename("value":abbr,inplace=True)
    
    data_Real_estate = data_Real_estate.pct_change() #change percentage
    #data_Real_estate[abbr]=(data_Real_estate[abbr]-data_Real_estate[abbr][0])/data_Real_estate[abbr][0]*100.0
    if main_data.empty:
        main_data=data_Real_estate
    else:
        main_data=main_data.join(data_Real_estate,lsuffix='_left', rsuffix='_right')
print(main_data)
main_data.to_csv("test3.csv")

#to save the data into bytestream
pickle_out=open("fiddy_state_pickel.pickle","wb")
pickle.dump(main_data,pickle_out)
pickle_out.close()

#To open the data from the pickle
pickle_input=open("fiddy_state_pickel.pickle","rb")
HPI_data=pickle.load(pickle_input)
pickle_input.close()
print(HPI_data)

#Using the pandas
main_data.to_pickle("pickel.pickle")
HPI_data=pd.read_pickle("pickel.pickle")
print(HPI_data)



HPI_data.plot()
plt.legend().remove()
plt.show()

Resmaple_data=HPI_data.resample("A",how='mean')
Resmaple_data.head()
HPI_data["TX"].plot()
Resmaple_data["TX"].plot()

Resmaple_data2=HPI_data["TX"].resample("A",how='ohlc')
print(Resmaple_data2)
#how sum,mean,ohlc
HPI_data.describe()
#correlation
print(HPI_data.corr())
print(HPI_data.corr().describe())
#----------------------------------------------------------------------
#Tutorail 9
#resample the data to year by mean,create the new column called TXY2
HPI_data["TXY2"]=HPI_data["TX"].resample("A",how='mean')
print(HPI_data[["TXY2","TX"]].head())

#Drop the rows that have null values
#HPI_data.dropna(inplace="TRUE")
#print(HPI_data[["TXY2","TX"]].head())

#Drop rows when all the are NaN
#HPI_data.dropna(how="all",inplace="TRUE")
#print(HPI_data[["TXY2","TX"]].head())

#axis=0 means remove the rows
#HPI_data.dropna(axis=0,how="all",inplace="TRUE")

#fillna ffill
HPI_data.fillna(method="ffill",inplace="TRUE")
print(HPI_data[["TXY2","TX"]])#.head())
#fillna bfill
HPI_data.fillna(method="bfill",inplace="TRUE")
print(HPI_data[["TXY2","TX"]])#.head())

#Also fill with some values
HPI_data.fillna(value=-99999,inplace="TRUE")
print(HPI_data[["TXY2","TX"]])#.head())

HPI_data.fillna(value=-99999,limit=10,inplace="TRUE")
print(HPI_data[["TXY2","TX"]])#.head())

#--------------------------
#tutorial11
HPI_data["TX12MA"]=pd.rolling_mean(HPI_data["TX"],12)
HPI_data.dropna(inplace="True")
print(HPI_data['TX12MA'])
#http://pandas.pydata.org/pandas-docs/version/0.15.2/computation.html#moving-rolling-statistics-moments
#---------------------------------------------------------------------
dict1={"col1":[20,23,np.nan],"col2":[23,34,45]}
df=pd.DataFrame(dict1)
print(df)
df.dropna(inplace="True")
print(df)



