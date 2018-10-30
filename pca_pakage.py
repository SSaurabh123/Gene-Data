
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
get_ipython().magic('matplotlib inline')

def perform_task(df_data,df_meta):
    if df_data.empty or df_meta.empty:
        print('Cannot perform pca')
    df_data = df_data.fillna(value=0, axis=1)
    df_data = df_data.T
    new_header = df_data.iloc[0]
    df_data = df_data[1:] 
    df_data.columns = new_header
    df_meta["Time-Unit"] = df_meta["Time"].map(str) + df_meta["Unit"]
    x = StandardScaler().fit_transform(df_data)   
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, df_meta[['Time-Unit']]], axis = 1)
    sns.set(color_codes=True)
    sns.lmplot( x="principal component 1", y="principal component 2",
    data=finalDf, 
    fit_reg=False, 
    hue='Time-Unit',
    legend=True,
    scatter_kws={"s": 80})

