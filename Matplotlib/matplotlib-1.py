#lesson1 and Lesson2
#Introduction to Matplotlib and basic line
#Legends, Titles, and Labels with Matplotlib
import matplotlib.pyplot as plt
x=[1,2,3,4,5,7]
y=[20,30,40,50,60,70]


x2=[1,2,3,4,5,7]
y2=[70, 60, 50, 40, 30, 20]

plt.plot(x,y,label="First Chart")
plt.plot(x2,y2,label="Second Chart")

plt.xlabel("x-lable")
plt.ylabel("y-lable")
plt.title("Testing Graph\nCheck it out")
plt.legend()

#plt.scatter(y,x)
plt.show()



