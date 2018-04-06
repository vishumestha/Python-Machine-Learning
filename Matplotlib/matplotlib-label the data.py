from matplotlib import pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

A = -0.75, -0.25, 0, 0.25, 0.5, 0.75, 1.0
B = 0.73, 0.97, 1.0, 0.97, 0.88, 0.73, 0.54

plt.plot(A,B)
for xy in zip(A, B):                                       # <--
    ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data') # <--

plt.grid()
plt.show()
plt.clf()

x=[1,2,3,4,5,6,7]
y=[30,2,18,13,60,54,36]

fig = plt.figure()
ax = fig.add_subplot(111)

plt.scatter(x,y,s=200,marker="*",color='#DC143C',label="Scatter")
for xy in zip(x, y):                                       # <--
    ax.annotate('(%s, %s)' % xy, xy=xy,textcoords='data') # <--

plt.xlabel("x-lable")
plt.ylabel("y-lable")
plt.title("Testing Graph\nCheck it out")

plt.legend()
plt.grid()
plt.show()

names=zip(A,B)
print(names)