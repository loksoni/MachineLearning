import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)

iris_dataframe = pd.DataFrame(X_train, columns=iris.feature_names)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
X_new = np.array([[5.6,3.0,4.1,1.3]])
prediction = knn.predict(X_new)
print("Predicted species:",iris['target_names'][prediction])
y_pred = knn.predict(X_test)
print("Test set score:",knn.score(X_test,y_test))