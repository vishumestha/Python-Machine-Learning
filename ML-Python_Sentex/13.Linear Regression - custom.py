

#The optimal values of m and b can be actually calculated with way less effort than doing a linear regression. 
#this is just to demonstrate gradient descent

from numpy import *
import matplotlib.pyplot as plt
from matplotlib import  style
style.use("ggplot")
from statistics import *
import pandas as pd
r_squared_for_each_gradient=[]

# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def coefficient_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]

    squared_error_regr = sum((ys_line - ys_orig) * (ys_line - ys_orig))
    squared_error_y_mean = sum((y_mean_line - ys_orig) * (y_mean_line - ys_orig))

    print(squared_error_regr)
    print(squared_error_y_mean)

    r_squared = 1 - (squared_error_regr/squared_error_y_mean)

    return r_squared

def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
        r_squared=compute_error_for_line_given_points(b, m, points)
        r_squared_for_each_gradient.append(r_squared)
        
    return [b, m]

def run():
    points=genfromtxt("D:/Data Science with Python/linear_regression_live-master/linear_regression_live-master/data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 1000
    
    
    print ("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print ("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print( "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))
    x=points[:,0]
    y=points[:,1]
    plt.scatter(x,y)
    regression_line=[m*x1+b for x1 in x]
    plt.plot(x,regression_line,color="r")
                 
    r_squared = coefficient_of_determination(y,regression_line)
    print(r_squared)
    
    iteration_line=[x for x in range(1,num_iterations+1)]
    #plt.scatter(iteration_line,r_squared_for_each_gradient)
    #print(df2=pd.DataFrame({"it":iteration_line,"gr":r_squared_for_each_gradient}))

    
    
    plt.plot(x,regression_line)
    plt.show()

if __name__ == '__main__':
    run()
