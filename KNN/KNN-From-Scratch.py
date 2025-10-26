import numpy as np
from collections import Counter
import pandas as pd
import random



class KNeighbours:
    """
    A class that can be used for classification with K Nearest Neighbours
    """
    def __init__(self, k=3):
        """
        

        Parameters
        ----------
        k : int, optional
            The default is 3. How many points we compare our point to

        Returns
        -------
        None.

        """
        self.k = k
    
    def fit(self, data, classification):
        """
        

        Parameters
        ----------
        data : np array or list
            Datapoints without their class, only the features
        classification : np array or list
            The classes of the datapoints. The classes should map up directly with the data

        Returns
        -------
        None. Creates variables inplace

        """
        self.data = np.array(data)
        self.classification = np.array(classification)
        
        
    def predict(self, point):
        """

        Parameters
        ----------
        point : np.array or list
            The point that you want to classify.

        Returns
        -------
        vote_result : str
            The class of your point as determined by the KNN algorithm.

        """
    
        distances = []
        for i in range(len(self.data)):
            euclidean_distance = np.linalg.norm(np.array(self.data[i]) - np.array(point))
            distances.append([euclidean_distance, self.classification[i]])
        
        votes = [i[1] for i in sorted(distances)[:self.k]]
        vote_result = Counter(votes).most_common(1)[0][0]
        
        return vote_result
    def score(self, X, y):
        """
        

        Parameters
        ----------
        X : np.array or list
            Datapoints without their class, only the features
        y : np.array or list
            The classes of the datapoints, what we want to predict. The classes should map up directly with the data.

        Returns
        -------
        accuracy : float
            The accuracy of our model in classifying the points

        """
        correct = 0
        total = 0
        for i in range(len(X)):
            vote = self.predict(X[i])
            if vote == y[i]:
                correct +=1
            total +=1
        self.accuracy = correct / total
        return self.accuracy


df = pd.read_csv(r"breast-cancer-wisconsin.data")
df.replace("?", -99999, inplace = True)
df.drop(["id"], axis = 1, inplace = True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.2
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

X_train = [i[:-1] for i in train_data]
y_train = [i[-1] for i in train_data]
X_test = [i[:-1] for i in test_data]
y_test = [i[-1] for i in test_data]

    
clf = KNeighbours(k = 20)
clf.fit(X_train, y_train)

print(f"The accuracy is {clf.score(X_test, y_test)}")
