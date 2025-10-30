import numpy as np
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"breast-cancer-wisconsin.data")

df.drop(["id"], axis = 1, inplace = True)
df.replace("?", -99999, inplace = True)

X = np.array(df.drop(["class"], axis = 1))
y = np.array(df["class"])

X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf  = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print(f"The accuracy of the SVM on this dataset is {accuracy}")