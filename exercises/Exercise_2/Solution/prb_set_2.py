import pandas as pd
import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt
##################
# exercise 1
##################
### a)
df = pd.read_csv('data/cars.csv')

X = df[['Weight','Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)

X_pred=pd.DataFrame(data={'Weight':[2300],'Volume':[1300]})
pred = regr.predict(X_pred)

### b)
print('The coefficients are', regr.coef_)
print('Which means that if the weigh increase by 1, the models predicts that the CO2 increases by ', regr.coef_[0])

### c)
X_pred=pd.DataFrame(data={'Weight':[2300+1000],'Volume':[1300]})
pred_2 = regr.predict(X_pred)

##################
# Exercise 2
##################

def func(x):
    f=np.array([
        5*x[0]**2 - x[1]**2,
        x[1]-0.25*(np.sin(x[0])+np.cos(x[1]))
    ])
    return f

def d_func(x):
    f=np.array([
        [10*x[0], 2*x[1]],
        [-0.25 * np.cos(x[0]), 1 + 0.25*np.sin(x[1])]
    ])
    return f


x = np.array([0.1, 0.1])
x_history=[x]
error_history = [np.mean(np.abs(func(x)))]
for i in range(10):
    x = x- func(x)@np.linalg.inv(d_func(x))

    x_history.append(x)
    error_history.append(np.mean(np.abs(func(x))))

plt.plot(error_history)
plt.ylabel('Average absolute error')
plt.xlabel('Iteration')
plt.show()

plt.plot(np.array(x_history)[:,0],label=r'$x_0$')
plt.plot(np.array(x_history)[:,1],label=r'$x_1$')
plt.ylabel('Coeff values')
plt.xlabel('Iteration')
plt.legend()
plt.show()

##################
# Exercise 3
##################

def func(x):
    return x**2 - 34
def d_func(x):
    return 2*x

x = 6.0
for i in range(10):
    x = x-func(x)/d_func(x)

print(f"Newtown's estimate {x}, numpy value {np.sqrt(34)}")