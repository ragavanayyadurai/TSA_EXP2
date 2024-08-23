### Developed by: Ragavendran A
### Register Number: 212222230114
### Date:

# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:

1. Import necessary libraries (NumPy, Matplotlib)
2. Load the dataset
3. Calculate the linear trend values using least square method
4. Calculate the polynomial trend values using least square method
5. End the program

### PROGRAM:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data= pd.read_csv("/content/Summer_olympic_Medals.csv")
data.head()
data['date']=pd.to_datetime(data['Year'])
data.info()
X=np.arange(len(data)).reshape(-1,1)
y=data['Gold'].values
```
A - LINEAR TREND ESTIMATION
```python
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X,y)
y_linear_predict=regressor.predict(X)


plt.figure(figsize=(10, 6))
plt.plot(data['date'], y, label='Actual Data', color='black')
plt.plot(data['date'], y_linear_predict, label='Linear Trend', color='blue')
plt.title('Medals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.legend()
plt.show()

print(f"Linear Trend Equation: y = {regressor.coef_[0]:.2f} * x + {regressor.intercept_:.2f}")
```

B - POLYNOMIAL TREND ESTIMATION:
```python
from sklearn.preprocessing import PolynomialFeatures

degree=2
poly_reg=PolynomialFeatures(degree=degree)
X_poly=poly_reg.fit_transform(X)
poly_reg.fit(X_poly,y)
regressor_poly=LinearRegression()
regressor_poly.fit(X_poly,y)
y_predict_poly_2=regressor_poly.predict(X_poly)


degree_3=3
poly_reg_3=PolynomialFeatures(degree=degree_3)  
X_poly_3=poly_reg_3.fit_transform(X)
poly_reg_3.fit(X_poly_3,y)
regressor_poly_3=LinearRegression()
regressor_poly_3.fit(X_poly_3,y)
y_predict_poly_3=regressor_poly_3.predict(X_poly_3)
```
```python
# Graph for polynomial trend
plt.figure(figsize=(10, 6))
plt.plot(data['date'], y, label='Actual Data', color='black')
plt.plot(data['date'], y_predict_poly_2, label=f'Polynomial Trend (Degree {degree})', linestyle='-.', color='green')
plt.plot(data['date'], y_predict_poly_3, label=f'Polynomial Trend (Degree {degree_3})',  color='red')
plt.title('Medals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.legend()
plt.show()


print("Polynomial Trend Equation (Degree 2): y = {:.2f} * x^2 + {:.2f} * x + {:.2f}".format(
    regressor_poly.coef_[2], regressor_poly.coef_[1], regressor_poly.intercept_))
print("Polynomial Trend Equation (Degree 3): y = {:.2f} * x^3 + {:.2f} * x^2 + {:.2f} * x + {:.2f}".format(
    regressor_poly_3.coef_[3], regressor_poly_3.coef_[2], regressor_poly_3.coef_[1], regressor_poly_3.intercept_))
```

### OUTPUT:

A - LINEAR TREND ESTIMATION:
![image](https://github.com/user-attachments/assets/51ff9d19-5091-430d-8c5f-bc2273dec630)


B- POLYNOMIAL TREND ESTIMATION:
![image](https://github.com/user-attachments/assets/fc57d30a-a1a6-43c3-bc1f-30429b580307)

### RESULT:
Thus the python program for linear and Polynomial Trend Estimation has been executed successfully.
