import matplotlib.dates as mdates
import numpy as np
import urllib
import matplotlib.pyplot as plt


def grpg_data(stock):
    stock_url='http://chartapi.finance.yahoo.com/instrument/1.0/'+stock+'/chartdata;type=quote;range=10y/csv'
    source_code=urllib.request.urlopen(stock_url).read().decode()
    source_code_line=source_code.split("\n")
    stock_data = []
    for line in source_code_line:
        source_code_line_split=source_code_line.split(",")
        if len(source_code_line_split)==6:
            if "values" not in source_code_line_split:
                stock_data.append(source_code_line_split)

grpg_data('TSLA')               