#lesson3 and Lesson4
#Bar Charts and Histograms with Matplotlib
import matplotlib.pyplot as plt
plt.clf()
x=[1,3,5,7,9,11]
y=[20,30,40,50,60,70]


x2=[2,4,6,8,10,12]
y2=[70, 60, 50, 40, 30, 20]

#Bar Chart
plt.bar(x,y,color='Orange',label="Bar Chart")
plt.bar(x2,y2,color='#DC143C',label="Bar Chart")
plt.legend()
plt.show()


#Histogram
plt.figure()
poppulation_age=[10,23,23,24,54,7,34,65,47,87,12,98,65,45,76,79,45,65,23,31,54,70,51,66,55,33,22,87]
plt.hist(poppulation_age,bins=10,histtype='bar',color='#DC143C',rwidth=0.8)

#bins=[10,20,30,40,50,60,70,80,90,100]
#plt.hist(poppulation_age,bins,histtype='bar',color='#DC143C',rwidth=0.8)

plt.xlabel("x-lable")
plt.ylabel("y-lable")
plt.title("Testing Graph\nCheck it out")
plt.legend()
plt.show()
plt.clf()

#lesson4
#Scatter Plots with Matplotlib

x=[1,2,3,4,5,6,7]
y=[30,2,18,13,60,54,36]
plt.scatter(x,y,s=200,marker="*",color='#DC143C',label="Scatter")

plt.xlabel("x-lable")
plt.ylabel("y-lable")
plt.title("Testing Graph\nCheck it out")
plt.legend()
plt.grid()
plt.show()







