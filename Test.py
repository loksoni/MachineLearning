import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.datasets
data = {
    'classes':[[1,2],[2,3],[3,4],[4,5],[5,6]],
    'labels':[0,0,0,1,1]
}
d=sklearn.datasets.base.Bunch(data=data['classes'], target=data['labels'])
print(d['data'])
X_tr, X_te, y_tr, y_te = train_test_split(d['data'],d['target'],random_state=0)
X_train = np.array(X_tr)
X_test = np.array(X_te)
y_train = np.array(y_tr)
y_test = np.array(y_te)
print("X_train:",X_train.shape)
print("X_test:",X_test.shape)

print("y_train:",y_train.shape)
print("y_test:",y_test.shape)
print(type(X_test))
