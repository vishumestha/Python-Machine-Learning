#lesson5
#Stack Plots with Matplotlib

import matplotlib.pyplot as plt

days = [1,2,3,4,5]

sleeping = [7,8,6,11,7]
eating =   [2,3,4,3,2]
working =  [7,8,7,2,2]
playing =  [8,5,7,8,13]

#Stackplot wont have legends so below is the alternative to display the legends
plt.plot([],[],label="sleeping",color='m',linewidth=5)
plt.plot([],[],label="sleeping",color='k',linewidth=5)
plt.plot([],[],label="sleeping",color='r',linewidth=5)
plt.plot([],[],label="sleeping",color='b',linewidth=5)


plt.stackplot(days,sleeping,eating,working,playing,colors=['m','k','r','b'])
plt.xlabel("x-lable")
plt.ylabel("y-lable")
plt.title("Testing Graph\nCheck it out")
plt.legend()
plt.show()