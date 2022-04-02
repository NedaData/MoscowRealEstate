# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 10:44:48 2022

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 12:04:32 2022

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
%matplotlib inline
path = r"C:\Users\user\Downloads\all_v2.csv\all_v2.csv"
df_moscow = pd.read_csv(path)
print(df_moscow.columns)
print(df_moscow[['region','building_type', 'levels', 'area'][0:5]])
print(df_moscow.dtypes)

df_moscow = df_moscow.drop(['date','time'], axis = 1)
print(df_moscow.dtypes)
df_moscow.drop_duplicates()
df_moscow1 = df_moscow[0:1000]
print(df_moscow1.shape)

X_data = df_moscow1['kitchen_area']
y_data = df_moscow1['price']

width = 12
height = 10

#plt.figure(figsize=(width, height))
sns.regplot(X_data, y_data, data = df_moscow1)
print(df_moscow1)

#sns.residplot(X_data, y_data, data = df_moscow1)

from scipy import stats
print(df_moscow1.describe())
pearson_coeff, p_value = stats.pearsonr(X_data, y_data)
print('Pearson coefficient is ',pearson_coeff, ' and P values is ',p_value)
print('Price max ',df_moscow1.max(), 'Price min ', df_moscow1.min())
#df_moscow1[['price','area']] = df_moscow1[['price','area']].astype("int")
print(df_moscow1.dtypes)

#The best score
grouped_test = df_moscow1[['price','building_type','area']].groupby(['building_type'])
#print(type(grouped_test))
#print(grouped_test.head(2))

#ANOVA
f_val, p_val = stats.f_oneway(grouped_test.get_group(2)['price'], grouped_test.get_group(4)['price'],grouped_test.get_group(3)['price'])
print('Fi value is ',f_val,' and P value is ',p_val)


f_val, p_val = stats.f_oneway(grouped_test.get_group(2)['price'], grouped_test.get_group(4)['price'])
print('Fi value is ',f_val,' and P value is ',p_val)

#Model Development

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
X = df_moscow1[['area']]
y = df_moscow1[['price']]
lm = LinearRegression()
lm.fit(X, y)
Yhat = lm.predict(X)
print(Yhat[0:5])

width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(X, y, data = df_moscow1)
plt.ylim(0,)

width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(X, y, data = df_moscow1)
plt.show()


print("Linear model intercepr ",lm.intercept_, " and linear model slope",lm.coef_)
#print("Linear model score is ",lm.score(X, y)," and mean squared error is ",mean_squared_error(y, Yhat))

from sklearn.preprocessing import PolynomialFeatures
Z = df_moscow1[['area','price','kitchen_area','levels']]
lm1 = LinearRegression()
lm1.fit(Z, y)
Yhat1 = lm1.predict(Z)
print(Yhat1[0:5])

plt.figure(figsize=(width,height))
ax1 = sns.distplot(y, hist = False, color = 'r', label = 'Fitted Values')
sns.distplot(Yhat1, hist = False, color = 'b', label = 'Fitted values', ax = ax1)
plt.title('Actual vs Fitted values for Price')
plt.xlabel('Price in rusian currency')
plt.ylabel('Area of real estate')
plt.show()
plt.close()
    
#print("MultiLinear model score is ",lm1.score(Z, y)," and mean squared error is ",mean_squared_error(y, Yhat1))

def PlotPoly(model, independent_variable, dependent_variable,Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)
    
    plt.plot(independent_variable, dependent_variable, '.', x_new, y_new, '-')
    plt.title('Polynomial fit with Mathplotlib for Price  near Length')
    ax = plt.gca()
    ax.set_facecolor((0.898,0.898,0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')
    
    plt.show()
    plt.close()

df_moscow1[['price','area']] = df_moscow1[['price','area']].astype('float')
print(df_moscow1.dtypes)

x = df_moscow1['area']
y = df_moscow1['price']

f = np.polyfit(x, y, 3)
p = np.poly1d(f)
PlotPoly(p, x, y, 'Area of Real Estates')

#Multiple Polynomial
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree=2)
Z_pr = pr.fit_transform(Z)
print(Z.shape)
print(Z_pr.shape)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
Input = [('scale',StandardScaler()),('polynomial',PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
pipe = Pipeline(Input)
Z = Z.astype('float')
pipe.fit(Z,y)
Ypipe = pipe.predict(Z)
print(Ypipe[0:5])

from sklearn.metrics import r2_score
r_squared = r2_score(y, Ypipe)

print("Linear model score is ",lm.score(X, y)," and mean squared error is ",mean_squared_error(y, Yhat))

print("MultipleLinear model score is ",lm1.score(Z, y)," and mean squared error is ",mean_squared_error(y, Yhat1))

print("Multipolynomial model score is ",r_squared," and mean squared error is ",mean_squared_error(y, Ypipe))

print(df_moscow1.columns)
print(df_moscow1['region'].value_counts())

X2 = df_moscow1[['price', 'geo_lat', 'geo_lon','building_type', 'level',
       'levels', 'rooms', 'area', 'kitchen_area', 'object_type']].values
Y2 = df_moscow1['region'].values

X2 = preprocessing.StandardScaler().fit(X2).fit_transform(X2.astype('float'))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X2, Y2, test_size=0.2, random_state=4)
print('Train set ',x_train, y_train)
print('Test set ',x_test, y_test)

from sklearn.neighbors import KNeighborsClassifier
k = 4
neigh = KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train)
print(neigh)

YhatClass = neigh.predict(x_test)
print(YhatClass[0:30])

from sklearn import metrics
print('Train set accuracy ', metrics.accuracy_score(y_train, neigh.predict(x_train)))
print('Test set accuracy ',metrics.accuracy_score(y_test, YhatClass))


import geopandas as gpd
locations = list(zip(df_moscow1['geo_lat'],df_moscow1['geo_lon']))
print(locations[0:5])
import folium
from folium.plugins import HeatMap
import webbrowser
m = folium.Map(location=[df_moscow1['geo_lat'].mean(),df_moscow1['geo_lon'].mean()], tiles='stamentoner', zoom_start=10, control_scale=True)
HeatMap(locations).add_to(m)
m.save('map_1.html')
webbrowser.open('map_1.html')

