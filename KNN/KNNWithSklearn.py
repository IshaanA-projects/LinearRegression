import numpy as np
import pandas as pd
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\Poonam\Documents\Projects\Machine learning\Machine learning with python\KNN\breast-cancer-wisconsin.data")
df.replace("?", -99999, inplace = True) # Replaces missing data with an outlier that has minimal effect
df.drop(["id"], axis = 1, inplace = True) # Removes the id column as this is irrelevant to the class

X = np.array(df.drop(["class"], axis = 1))
y = np.array(df["class"])  # Creates feature and labels as arrays

X = preprocessing.scale(X) # Scales the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print(f"The accuracy of the algorithm is {accuracy}")

example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 1, 1, 3, 2, 1, 3, 1, 1]]) # Example data created to test model's prediction
example_measures= example_measures.reshape(len(example_measures), -1)
example_measures = preprocessing.scale(example_measures)

print(f"The prediction on our example data is {clf.predict(example_measures)}")