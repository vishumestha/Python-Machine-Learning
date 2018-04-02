import pandas as pd
from numpy import *
import matplotlib.pyplot as plt

def mean_squared_error(points,thetha0,thetha1):
    total_error=0
    M=len(points)
    for i in range(M):
        x=points[i,0]
        y=points[i,1]
        total_error=total_error+(((thetha0+x*thetha1)-y)**2)
    return total_error/M
 
#uisng numpy
#def mean_squared_error(points,thetha0,thetha1):
#    #np_points=np.array(points)
#    M=len(points)
#    x=points[:,0]
#    y=points[:,1]
#    error=((thetha0+x*thetha1)-y)**2
#    return np.sum(error)/M
#    
def step_wise_function(points,learning_rate,new_thetha0,new_thetha1):
    thetha0_gradient=0
    thetha1_gradient=0
    M=len(points)
    
    for i in range(len(points)):
        x=points[i,0]
        y=points[i,1]
        thetha0_gradient=thetha0_gradient+((new_thetha0+x*new_thetha1)-y)
        thetha1_gradient=thetha1_gradient+((new_thetha0+x*new_thetha1)-y)*x
    thetha0_gradient=new_thetha0-(learning_rate*thetha0_gradient)*(2/M)
    thetha1_gradient=new_thetha1-(learning_rate*thetha1_gradient)*(2/M)
    return thetha0_gradient,thetha1_gradient
    
def gradient_of_cost_funtion(points,learning_rate,thetha0,thetha1,num_iterations):
    new_thetha0=thetha0
    new_thetha1=thetha1
    list_error=[]
    for i in range(num_iterations):
        new_thetha0,new_thetha1=step_wise_function(points,learning_rate,new_thetha0,new_thetha1)
        error=mean_squared_error(points,new_thetha0,new_thetha1)
        list_error.append(error)
       
    return new_thetha0,new_thetha1,list_error
  
def run():
    
    points=genfromtxt("D:/data.csv",delimiter=',')
    df=pd.read_csv("D:/data.csv",names=["hours","score"])

    
    learning_rate=0.0001
    thetha0=0
    thetha1=0
    num_iterations=10
    error=mean_squared_error(points,thetha0,thetha1)
    print("before")
    print(error)
    thetha0,thetha1,list_error=gradient_of_cost_funtion(array(points),learning_rate,thetha0,thetha1,num_iterations)
    error=mean_squared_error(points,thetha0,thetha1)
    #hx=thetha0+x*thetha1
    hx=[]
    

    for i in range(len(points)):
       
        x=points[i,0]
        #print(i,x,thetha0,thetha1)
        hx.append(thetha0+(x*thetha1))
    plt.scatter(df["hours"],df["score"])
    plt.plot(df["hours"],hx)
    
    plt.show()
    
    plt.figure()
    index=[x for x in range(num_iterations)]
    print(len(index),len(list_error))
    plt.plot(index,list_error)
    
    plt.show()
    print("After")
    print(error)
    print(thetha0,thetha1)
    plt.show()


if __name__=="__main__":
    run()

