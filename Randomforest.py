import pandas as pd
import numpy as np
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
data = pd.read_csv('E:/Student/Assessment/Iris.csv')

x=data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y=data['Species']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
model = RandomForestClassifier()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print('accuracy score  is', accuracy_score(y_test,y_pred)*100,'%')

