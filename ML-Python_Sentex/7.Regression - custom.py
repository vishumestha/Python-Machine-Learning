from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("fivethirtyeight") 
xs=np.array([1,2,3,4,5],dtype=np.float64)
ys=np.array([5,4,6,5,6],dtype=np.float64)

#

def best_fit_line_and_slope(xs,ys):
    m=( mean(xs)*mean(ys)-mean(xs*ys) )/( (mean(xs)**2)-(mean(xs*xs)))
    b= mean(ys)-(m*mean(xs))
    
    return m,b
m,b= best_fit_line_and_slope(xs,ys)

regression_line=[(m*x)+b for x in xs]
plt.scatter(xs,ys,label="Actual")
plt.plot(xs,regression_line,label="Regression line",color="r")


predict_x=8
predicted_y=(m*predict_x)+b
plt.scatter(predict_x,predicted_y,color='g',label="Predicted")
plt.legend()
plt.show()

def squarederror(ys_orgin,ys_line):
    return(sum ((ys_orgin-ys_line) * (ys_orgin-ys_line)))



def coeffient_of_dertmination_cal(ys_orgin,ys_line):
    y_mean=[mean(ys_orgin) for y in ys_orgin]
    squared_error_regression=squarederror(ys_orgin,ys_line)
    #squared_error_y_mean=squarederror(y_mean,ys_line)
    squared_error_y_mean=squarederror(ys_orgin,y_mean)
    
    return (1-(squared_error_regression/squared_error_y_mean))
    
coeffient_of_dertmination=coeffient_of_dertmination_cal(ys,regression_line)

print(coeffient_of_dertmination)
                 