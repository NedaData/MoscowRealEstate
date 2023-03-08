#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
import geopandas as gpd
import tensorflow as tf
import requests
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
import folium
from folium.plugins import HeatMap
import webbrowser


# In[54]:


def input_data(url, path):
     if url is None:
        df = pd.read_csv(path)
        return df
     else:
        r = requests.get(url)
        open('open.csv', 'wb').write(r.content)
        df = pd.read_csv('open.csv')
        return df


# In[55]:


def hist_plot(x, y):
    
    N_points = 100000
    n_bins = 20

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    # We can set the number of bins with the `bins` kwarg
    axs[0].hist(x, bins=n_bins)
    axs[1].hist(y, bins=n_bins)


# In[56]:


def preprocessing(datasets):
    datasets = datasets.dropna()
    for x in datasets.columns.values:
        if x =='price_mill_rub':
            datasets = datasets.rename(columns={'price_mill_rub':'price'})
            datasets = pd.DataFrame(datasets, columns=['total_area','minutes','views','living_area','kitchen_area','price'])
            datasets = MinMaxScaler().fit_transform(df)
            datasets = pd.DataFrame(datasets, columns=['total_area','minutes','views','living_area','kitchen_area','price'])
        else:
            datasets = pd.DataFrame(datasets, columns=['total_area','minutes','views','living_area','kitchen_area','price'])
            datasets = MinMaxScaler().fit_transform(df)
            datasets = pd.DataFrame(datasets, columns=['total_area','minutes','views','living_area','kitchen_area','price'])
    return datasets


# In[57]:


def model_selection(datasets):
    X = df.drop(['price'], axis=1) #features
    y = df['price']
    X1 = df.drop(['price'], axis=1) #features1
    y1 = df['price']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state = 42)
    return X_train, X_test, y_train, y_test, 


# In[58]:


def Model0():
    X_train, X_test, y_train, y_test, X_train1, X_test1, y_train1, y_test1 = model_selection(datasets)
    
    tf.keras.backend.clear_session()
    
    #Model 0 with 0 datasets
    
    tf.random.set_seed(60)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, input_dim = X_train.shape[1], activation = 'relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, input_dim = X_train.shape[1], activation = 'relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=256, activation = 'relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=256, activation = 'relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=128, activation = 'relu'),
        tf.keras.layers.Dense(units=1, activation = 'relu'),

    ], name = 'Batchnorm',)
    plot_model(model, show_shapes = True)

    optimizer = tf.keras.optimizers.Adam(lr = 0.001)
    model.compile(loss = tf.keras.losses.mae,
                 optimizer = optimizer,
                 metrics = ['mae'])
    history = model.fit(X_train, y_train, epochs = 200, batch_size=1024, validation_data=(X_test, y_test), verbose=1)
    
    #Model 0 with 1 datasets
    
    tf.random.set_seed(60)
    model1 = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, input_dim = X_train1.shape[1], activation = 'relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, input_dim = X_train1.shape[1], activation = 'relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=256, activation = 'relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=256, activation = 'relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=128, activation = 'relu'),
        tf.keras.layers.Dense(units=1, activation = 'relu'),

    ], name = 'Batchnorm',)
    plot_model(model1, show_shapes = True)

    optimizer1 = tf.keras.optimizers.Adam(lr = 0.001)
    model1.compile(loss = tf.keras.losses.mae,
                 optimizer = optimizer1,
                 metrics = ['mae'])
    history1 = model1.fit(X_train1, y_train1, epochs = 200, batch_size=1024, validation_data=(X_test1, y_test1), verbose=1)
    
    #Plot Model 0 with datasets 0
    pd.DataFrame(history.history).plot();
    pd.DataFrame(history1.history).plot();
    
    score = model.evaluate(X_test, y_test)
    score1 = model1.evaluate(X_test, y_test)
    diff_score = abs(score1-score)
    return score, score1, diff_score
    


# In[59]:


def Model1():
    X_train, X_test, y_train, y_test, X_train1, X_test1, y_train1, y_test1 = model_selection(datasets)
    
    tf.keras.backend.clear_session()
    
    #Model 0 with 0 datasets
    
        
    tf.random.set_seed(60)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, input_dim = X_train.shape[1]),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, input_dim = X_train.shape[1]),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=256),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=256),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=128),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(units=1, activation = "linear"),

    ], name = 'Batchnorm',)
    plot_model(model, show_shapes = True)

    optimizer = tf.keras.optimizers.Adam(lr = 0.001)
    model.compile(loss = tf.keras.losses.mae,
                 optimizer = optimizer,
                 metrics = ['mae'])
    history = model.fit(X_train, y_train, epochs = 200, batch_size=1024, validation_data=(X_test, y_test), verbose=1)
    
    #Model 0 with 1 datasets
    
    tf.random.set_seed(60)
    tf.keras.backend.clear_session()
    model1 = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, input_dim = X_train1.shape[1]),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, input_dim = X_train1.shape[1]),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=256),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=256),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=128),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(units=1, activation = "linear"),

    ], name = 'Batchnorm',)
    plot_model(model1, show_shapes = True)

    optimizer1 = tf.keras.optimizers.Adam(lr = 0.001)
    model1.compile(loss = tf.keras.losses.mae,
                 optimizer = optimizer1,
                 metrics = ['mae'])
    history1 = model1.fit(X_train1, y_train1, epochs = 200, batch_size=1024, validation_data=(X_test1, y_test1), verbose=1)
    
    #Plot Model 0 with datasets 0
    pd.DataFrame(history.history).plot();
    pd.DataFrame(history1.history).plot();
    
    score = model.evaluate(X_test, y_test)
    score1 = model1.evaluate(X_test1, y_test1)
    diff_score = abs(score1-score)
    return score, score1, diff_score


# In[60]:


def Model2():
    X_train, X_test, y_train, y_test, X_train1, X_test1, y_train1, y_test1 = model_selection(datasets)
    
    tf.keras.backend.clear_session()
    
    #Model 0 with 0 datasets
    
        
    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1024, input_dim = X_train.shape[1]),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Dense(512),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Dense(512),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Dense(256),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Dense(256),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Dense(128),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(units=1, activation = 'linear'),
    
    ], name = "Larger_network",)
    plot_model(model, show_shapes = True)

    optimizer = tf.keras.optimizers.Adam(lr = 0.001)
    model.compile(loss = tf.keras.losses.mae,
                 optimizer = optimizer,
                 metrics = ['mae'])
    history = model.fit(X_train, y_train, epochs = 200, batch_size=1024, validation_data=(X_test, y_test), verbose=1)
    
    #Model 0 with 1 datasets
    
    tf.random.set_seed(60)
    tf.keras.backend.clear_session()
    model1 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1024, input_dim = X_train.shape[1]),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Dense(512),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Dense(512),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Dense(256),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Dense(256),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Dense(128),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(units=1, activation = 'linear'),
    
    ], name = "Larger_network",)
    plot_model(model1, show_shapes = True)

    optimizer1 = tf.keras.optimizers.Adam(lr = 0.001)
    model1.compile(loss = tf.keras.losses.mae,
                 optimizer = optimizer1,
                 metrics = ['mae'])
    history1 = model1.fit(X_train1, y_train1, epochs = 200, batch_size=1024, validation_data=(X_test1, y_test1), verbose=1)
    
    #Plot Model 0 with datasets 0
    pd.DataFrame(history.history).plot();
    pd.DataFrame(history1.history).plot();
    
    score = model.evaluate(X_test, y_test)
    score1 = model1.evaluate(X_test1, y_test1)
    diff_score = abs(score1-score)
    return score, score1, diff_score


# In[61]:


def Model3():
    X_train, X_test, y_train, y_test, X_train1, X_test1, y_train1, y_test1 = model_selection(datasets)
    checkpoint_name = 'Weights\Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
    callbacks_list = [checkpoint]
    tf.keras.backend.clear_session()
    
    #Model 0 with 0 datasets
    
        
    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1024, input_dim = X_train.shape[1]),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Dense(512),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Dense(512),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Dense(256),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Dense(256),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Dense(128),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(units=1, activation = 'linear'),
    
    ], name = "Larger_network",)

    plot_model(model, show_shapes = True)

    optimizer = tf.keras.optimizers.Adam(lr = 0.001)
    model.compile(loss = tf.keras.losses.mae,
                 optimizer = optimizer,
                 metrics = ['mae'])
    history = model.fit(X_train, y_train, epochs = 200, batch_size=1024, validation_data=(X_test, y_test), verbose=1)
    
    #Model 0 with 1 datasets
    
    tf.random.set_seed(60)
    tf.keras.backend.clear_session()
    model1 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1024, input_dim = X_train.shape[1]),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Dense(512),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Dense(512),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Dense(256),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Dense(256),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Dense(128),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(units=1, activation = 'linear'),
    
    ], name = "Larger_network",)

    plot_model(model1, show_shapes = True)

    optimizer1 = tf.keras.optimizers.Adam(lr = 0.001)
    model1.compile(loss = tf.keras.losses.mae,
                 optimizer = optimizer1,
                 metrics = ['mae'])
    history1 = model1.fit(X_train1, y_train1, epochs = 200, batch_size=1024, validation_data=(X_test1, y_test1), verbose=1)
    
    #Plot Model 0 with datasets 0
    pd.DataFrame(history.history).plot();
    pd.DataFrame(history1.history).plot();
    
    score = model.evaluate(X_test, y_test)
    score1 = model1.evaluate(X_test1, y_test1)
    diff_score = abs(score1-score)
    return score, score1, diff_score


# In[62]:


def final_result(Model0, Model1, Model2, Model3):
    score0, score10, diff_score0 = Model0()
    score1, score11, diff_score1 = Model1()
    score2, score12, diff_score2 = Model2()
    score3, score13, diff_score3 = Model3()
    list = [[dif_score0, dif_score1, dif_score2, dif_score3]]
    result = pd.DataFrame(list, columns=['Diff_Score One', 'Diff_Score Two', 'Diff_Score Three', 'Diff_Score Four'])
    return result


# In[63]:


def geolocation(datasets):
    if datasets['geo_lat'] is not None and datasets['geo_lon'] is not None:
        locations = list(zip(df_moscow1['geo_lat'],df_moscow1['geo_lon']))
        m = folium.Map(location=[df_moscow1['geo_lat'].mean(),df_moscow1['geo_lon'].mean()], tiles='stamentoner', zoom_start=10, control_scale=True)
        HeatMap(locations).add_to(m)
        m.save('map_1.html')
        webbrowser.open('map_1.html')
        return locations
    else:
        
        geo = geocode(data['addr'], provider='nominatim',user_agent='moscow_geo', timeout=4)
        return geo
        

