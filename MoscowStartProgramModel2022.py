# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 14:26:35 2022

@author: user
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import mean_squared_error
%matplotlib inline

path = r'C:\Users\user\Downloads\archive\moscow_real_estate_sale.csv'
df = pd.read_csv(path, delimiter = ',')
print(df.columns)
print(df.dtypes)
print(df.head())
df.drop(columns = ['Unnamed: 0'], axis = 1, inplace=True)
print(df.head())
#%matplotlib qt
#plt.plot(df['price'])
df1 = df[0:1000]
print(df1.shape)
A = df1[['total_area','minutes','views','living_area','kitchen_area']]
y = df1['price']
width = 12
height = 10
def plotReg(A, y):
    
    plt.figure(figsize=(width,height))
    sns.set_theme(style = 'darkgrid')
    for i in A:
        sns.jointplot(data = df1, y = y, x = i, kind= 'reg', truncate = False, color = 'm')
#plotReg(X,y)
def plotResid(A,y):
    
    plt.figure(figsize=(width,height))
    sns.set_theme(style='white')
    for i in A:
        sns.residplot(i, y, data = df1)
#plotResid(A, y)   
#df1.drop_duplicates()

#Descriptive statistic
from scipy import stats
print(df1.describe())
X = df1['total_area']
pearson_coef, p_value = stats.pearsonr(X,y)
print('Pearson coef ',pearson_coef,' P_value ',p_value)
print(df1.columns)
print(df1.dtypes)

X = df1[['total_area']]
from sklearn.linear_model import LinearRegression
Z = A

lr = LinearRegression()
lr.fit(X,y)
Yhat = lr.predict(X)
print("Prediction of linear model is ", Yhat[0:5])
intercept = lr.intercept_
coefficient = lr.coef_

score1 = lr.score(X, y)
mse1 = mean_squared_error(y, Yhat)
#print('Score ',score,' mse ',mse1)
#Multiplelinear model
from sklearn.preprocessing import PolynomialFeatures

lr.fit(Z, y)
Yhat1 = lr.predict(Z)
print('Prediction of multiplelinear model is ', Yhat1[0:5])

score2 = lr.score(Z, y)
mse2 = mean_squared_error(y, Yhat1)
#print('Score ',score1,' mse ',mse2)

plt.figure(figsize=(width, height))
ax1 = sns.distplot(df1['price'], hist = False, color = 'r', label = 'Actual values')
sns.distplot(Yhat1, hist = False, color = 'b', label = 'Fitted value', ax = ax1)
plt.title('Visualization of multiplelinear regression')
plt.xlabel('Proportion in areas')
plt.ylabel('Proportion in price')

plt.show()
plt.close()

def PlotPoly(model, independent_variable, dependent_variable,Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)
    
    plt.plot(independent_variable, dependent_variable, '.', x_new, y_new, '-')
    plt.title('Polynomial fit with Mathplotlib for Price')
    ax = plt.gca()
    ax.set_facecolor((0.898,0.898,0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Real State')
    
    plt.show()
    plt.close()
x = df1['total_area']
#Polynomial 1D
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)
PlotPoly(p, x, y, 'Total area')

#MultiplePolynomial 
pr = PolynomialFeatures(degree=2)
Z_pr = pr.fit_transform(Z)
print(Z.shape)
print(Z_pr.shape)

#Pipeline
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
Input = [('scale',StandardScaler()),('polynomial',PolynomialFeatures()),('model',LinearRegression())]
pipe = Pipeline(Input)
print(pipe)
Z = Z.astype(float)
pipe.fit(Z, y)
Yhat2 = pipe.predict(Z)
print(Yhat2[0:10])

r2_squared = r2_score(y, Yhat2)
mse3 = mean_squared_error(y, Yhat2)
print('Score for Simple Linear Regression ',score1,' and mean squared error is ',mse1)
print('Score for Multiple Linear Regression ',score2,' and mean squared error is ',mse2)
print('Score for Multiple Polynomial Linear Regression ',r2_squared,' and mean squared error is ',mse3)

#############################
#Evaluation
df1 = df1._get_numeric_data()
print(df1.head())
from ipywidgets import interact, interactive, fixed, interact_manual

def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in Rusian currency)')
    plt.ylabel('Proportion of Real Estate')

    plt.show()
    plt.close()
def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    
    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 
 
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)


    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()
    
from sklearn.model_selection import train_test_split

y = df1['price']
x = df1.drop('price',axis=1)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1, random_state=1)

print('Number of test samples ',x_test.shape[0])
print('Number of training samples ',x_train.shape[0])

lre = LinearRegression()
lre.fit(x_train[['total_area']],y_train)
lre.score(x_test[['total_area']],y_test)

from sklearn.model_selection import cross_val_score

Rcross = cross_val_score(lre, x[['total_area']], y, cv=4)
print('Mean of fold are ',Rcross.mean(),' and standard deviation is ',Rcross.std())


print(-1 * cross_val_score(lre,x[['total_area']], y,cv=4,scoring='neg_mean_squared_error'))

from sklearn.model_selection import cross_val_predict
from sklearn.cluster import KMeans
YhatCross = cross_val_predict(lre, x[['total_area']], y, cv= 4)
print(YhatCross[0:5])

