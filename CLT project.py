import pandas as pd
import numpy as np
import os
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics    
from scipy import stats   
from scipy.stats import iqr
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

import statsmodels.api as sm


os.chdir("C:\\Users\luna\Temp")    

data = pd.read_csv('clt-runway-sample.csv')     
data = data.drop(['apt','man','rfz','sus','eng','dfx','sfx', 'a10', 'tcr'], axis = 1)

data.isnull().sum()


print("More than 2 nulls per row", (data.isnull().sum(1)>2).sum())
print("More than 3 nulls per row", (data.isnull().sum(1)>3).sum())
print("More than 4 nulls per row", (data.isnull().sum(1)>4).sum())


def ConvertToKnots(value):
    if value == 0:
        return
    returnValue = value
    if value < 1:
        returnValue = round((666 * value),0)
    return returnValue


data['bcn'].fillna(data['bcn'].mode()[0], inplace=True)
data['ina'].fillna(data['ina'].mean(), inplace=True)
#
data['ara'].replace('VFR', np.nan, inplace=True)  
data['ara'] = data['ara'].astype(float) 
data['ara'].fillna(data['ara'].quantile(.25), inplace=True)

data['spd'] = data['spd'].apply(ConvertToKnots)
data['spd'].fillna(data['spd'].mean(), inplace=True)

data['dap'].fillna(data['dap'].mode()[0], inplace=True)
data['mfx'].fillna(data['mfx'].mode()[0], inplace=True)
data['gat'].fillna(data['gat'].mode()[0], inplace=True)
data['scn'].fillna(data['scn'].mode()[0], inplace=True)
data['typ'].fillna(data['typ'].mode()[0], inplace=True)
data['sfz'].fillna(data['sfz'].mode()[0], inplace=True)
data['tds'].fillna(data['tds'].mode()[0], inplace=True)
data['cfx'].fillna(data['cfx'].mode()[0], inplace=True)
data['est'].fillna(data['est'].mode()[0], inplace=True)
data['oma'].fillna(data['oma'].mode()[0], inplace=True)
data['ooa'].fillna(data['ooa'].mode()[0], inplace=True)
data['o3a'].fillna(data['o3a'].mode()[0], inplace=True)


data["trw"] = data["trw"].astype('category').cat.codes    
data["dap"] = data["dap"].astype('category').cat.codes
data["mfx"] = data["mfx"].astype('category').cat.codes
data["gat"] = data["gat"].astype('category').cat.codes    
data["scn"] = data["scn"].astype('category').cat.codes    
data["typ"] = data["typ"].astype('category').cat.codes    
data["sfz"] = data["sfz"].astype('category').cat.codes    
data["tds"] = data["tds"].astype('category').cat.codes    
data["cfx"] = data["cfx"].astype('category').cat.codes    
data["est"] = data["est"].astype('category').cat.codes    
data["oma"] = data["oma"].astype('category').cat.codes    
data["ooa"] = data["ooa"].astype('category').cat.codes    
data["o3a"] = data["o3a"].astype('category').cat.codes    
data["cfg"] = data["cfg"].astype('category').cat.codes    


def getOutliers(data, name):
    outliers = []
    lower = data.quantile(.25)
    upper = data.quantile(.75)
    iqrange = upper - lower
    lowerValue = lower - (1.5 * iqrange)
    upperValue = upper + (1.5 * iqrange)
    outliers = [x for x in data if x < lowerValue or x > upperValue ]
    print ("Outlier summary for:" , name)
    print ("Total outliers:", len(outliers))
    print (np.sum(outliers < lowerValue), "outliers below", lowerValue, " calculated by", lower, "- ( 1.5 *", iqrange, ")"  )
    print (np.sum(outliers > upperValue), "outliers above", upperValue, " calculated by", upper, "+ ( 1.5 *", iqrange, ")"  )
    return outliers

ara = getOutliers(data['ara'], "ara")
spd = getOutliers(data['spd'], "spd")

def ReplaceValues(value, limit, upperOrLower):
    if upperOrLower == "upper":
        if value >= limit:

            return limit
    else:
        if value <= limit:

            return limit
    return value

data['ara'] = data['ara'].apply(lambda x: ReplaceValues(x, 42000, "upper") )
data['spd'] = data['spd'].apply(lambda x: ReplaceValues(x, 504, "upper") )
data['spd'] = data['spd'].apply(lambda x: ReplaceValues(x, 392, "lower") )

print(data['ara'].describe())
print(data['spd'].describe())

print("after imputations",data.isnull().sum())


columns=['cfx','dap','mfx','gat','sfz','tds','bcn','scn','ara','oma','ooa','o3a','spd','typ','est','ina','cfg','trw']
winnowed_dataset=data[columns]
winnowed_dataset.to_csv('new_clean_dataset.csv',index=False) 

training_features = ['cfx','dap','mfx','gat','sfz','tds','bcn','scn','ara','oma','ooa','o3a','spd','typ','est','ina','cfg']
target = 'trw'
X = data[training_features]
y = data[target]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)
logreg = LogisticRegression()


logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)


y_pred_proba = logreg.predict_proba(X_test)[::,1]           
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
print("AUC:",round(auc,4))


logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary())


import pandas as pd
clt_df = pd.read_csv("/temp/new_clean_dataset.csv")
clt_df2 = pd.read_csv("new_clean_dataset.csv")
print(pd.DataFrame.equals(clt_df, clt_df2))
print(clt_df2.head(5))
print(clt_df2.shape)
num_of_classes=len(clt_df2.gat.unique())
print(num_of_classes)
clt_df2.describe()
x=clt_df2.drop(axis=0, columns=['gat','trw'])
y=clt_df2.gat
print(x.shape)
print(y.shape)
from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test=train_test_split(x,y,test_size=0.33,random_state=42)
from sklearn import svm
frp=svm.SVC(kernel='linear')
frp.fit(x_train,y_train)
y_pred=frp.predict(x_test)

from sklearn import metrics
print("Accuracy:" , metrics.accuracy_score(y_test, y_pred))
print("Precision:" , metrics.accuracy_score(y_test, y_pred))
print("Recall:" , metrics.accuracy_score(y_test, y_pred))



