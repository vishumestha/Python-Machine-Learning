import pandas as pd
from numpy import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression,SGDClassifier,SGDRegressor
   
def mean_squared_error(points,thetha0,thetha1,thetha2):
    total_error=0
    M=len(points)
    for i in range(len(points)):
        x=points[i,0]
        z=points[i,1]
        y=points[i,2]
        hx=sigmoid_function(thetha0,thetha1,thetha2,x,z)
        
        total_error=total_error+((y*np.log(hx))+((1-y)*np.log(1-hx)))
    return -(total_error/M)
    
def sigmoid_function(new_thetha0,new_thetha1,new_thetha2,x,z):
    
    thetha=new_thetha0+x*new_thetha1+z*new_thetha2
    h_x=1/(1+np.exp(-thetha))
    
    return h_x
    
 
def step_wise_function(points,learning_rate,new_thetha0,new_thetha1,new_thetha2):
    thetha0_gradient=0
    thetha1_gradient=0
    thetha2_gradient=0
    h_x=0
    M=len(points)
    
    for i in range(len(points)):
        x=points[i,0]
        z=points[i,1]
        y=points[i,2]
        h_x=sigmoid_function(new_thetha0,new_thetha1,new_thetha2,x,z)
        thetha0_gradient=thetha0_gradient+((h_x-y))
        thetha1_gradient=thetha1_gradient+(h_x-y)*x
        thetha2_gradient=thetha2_gradient+(h_x-y)*z

    thetha0_gradient=new_thetha0-(learning_rate*thetha0_gradient)*(2/M)
    thetha1_gradient=new_thetha1-(learning_rate*thetha1_gradient)*(2/M)
    thetha2_gradient=new_thetha2-(learning_rate*thetha2_gradient)*(2/M)
    
    return thetha0_gradient,thetha1_gradient,thetha2_gradient
    
def gradient_of_cost_funtion(points,learning_rate,thetha0,thetha1,thetha2,num_iterations):
    new_thetha0=thetha0
    new_thetha1=thetha1
    new_thetha2=thetha2
    list_error=[]
    for i in range(num_iterations):
        new_thetha0,new_thetha1,new_thetha2=step_wise_function(points,learning_rate,new_thetha0,new_thetha1,new_thetha2)
        error=mean_squared_error(points,new_thetha0,new_thetha1,thetha2)
        list_error.append(error)
        
    return new_thetha0,new_thetha1,new_thetha2,list_error
  
def run():
    
    points=genfromtxt("D:/Data Science with Python/ex2data1.txt",delimiter=',')
    df=pd.read_csv("D:/Data Science with Python/ex2data1.txt",names=["hours","score","label"])

    
    
    learning_rate=0.0001
    thetha0=0
    thetha1=0
    thetha2=0
    num_iterations=70000
    error=mean_squared_error(points,thetha0,thetha1,thetha2)
    print("before")
    print(error)
    thetha0,thetha1,thetha2,list_error=gradient_of_cost_funtion(array(points),learning_rate,thetha0,thetha1,thetha2,num_iterations)
    error=mean_squared_error(points,thetha0,thetha1,thetha2)
    print("after")
    print(error)
    
    #using sklearn
#    model=SGDClassifier()
#    X=df[["hours","score"]]
#    y=df['label']
#    model.fit(X,y)
#    print(model.intercept_,model.coef_)
    #hx=thetha0+x*thetha1
    
    
    
    
    
    hx=[]
    
    for i in range(len(points)):
       
        x=points[i,0]
        z=points[i,1]
        y=points[i,2]
        #print(i,x,thetha0,thetha1)
        hx_fun=sigmoid_function(thetha0,thetha1,thetha2,x,z)
        #hx.append(hx_fun)
        #print(y,hx_fun)
        if hx_fun>=0.7:
            print(y,1,hx_fun)
            hx.append(1)
        else:
            print(y,0,hx_fun)
            hx.append(0)
    
    df['predict']=hx
    positive = df[df['label'].isin([1])]  
    negative = df[df['label'].isin([0])]
    fig, ax = plt.subplots(figsize=(12,8))  
    ax.scatter(positive["score"],positive["hours"], s=50, c='b', marker='o', label='Admitted')  
    ax.scatter(negative["score"],negative["hours"], s=50, c='r', marker='x', label='Not Admitted')  
    ax.legend()  
    
    #plt.plot(df["hours"],df["score"],hx)
    
    plt.show()
    
    plt.figure()
    index=[x for x in range(num_iterations)]
    #print(len(index),len(list_error))
    plt.plot(index,list_error)
    
    count=0
    y=df['label']
    for x,y in zip(y,hx):
        if x==y:
            count=count+1
    print(count)
    print(count/len(df['label']))
        
    #    plt.show()
#    print("After")
#    print(error)
    print(thetha0,thetha1,thetha2)
    plt.show()


if __name__=="__main__":
    run()

from sklearn.ensemble import RandomForestClassifier
import numpy as np
test=np.arange(100,1000,100)