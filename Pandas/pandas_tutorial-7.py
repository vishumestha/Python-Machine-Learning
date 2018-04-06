import pandas as pd
import quandl
import pickle
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
    data_Real_estate.columns=[abbr]
    #data_Real_estate.rename("value":abbr,inplace=True)
    if main_data.empty:
        main_data=data_Real_estate
    else:
        main_data=main_data.join(data_Real_estate,lsuffix='_left', rsuffix='_right')
print(main_data)
main_data.to_csv("test.csv")

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




