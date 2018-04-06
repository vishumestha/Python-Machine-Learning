import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

bridge_height = {'meters':[10.26, 10.31, 10.27, 10.22, 10.23, 6212.42, 10.28, 10.25, 10.31]}
df = pd.DataFrame(bridge_height)

df.plot()
plt.show()

df2=pd.rolling_mean(df['meters'],2)
test=df.describe()['meters']['std']
df=df[df['meters']<test]

