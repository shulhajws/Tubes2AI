  
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
import numpy as np
from collections import Counter

class sKNNClassifier:
    def __init__(self, k=3, distance_metric='euclidean', p=3):
        """
        Initialize the KNN classifier.
        Args:
        - k: Number of nearest neighbors.
        - distance_metric: Distance metric to use ('euclidean', 'manhattan', 'minkowski').
        - p: Parameter for Minkowski distance (default: 2).
        """
        self.k = k
        self.distance_metric = distance_metric
        self.p = p

    def fit(self, x_train, y_train):
        """
        Store training data.
        Args:
        - x_train: Training data features (2D array).
        - y_train: Multi-target labels (2D array, each row contains [label, attack_cat]).
        """
        self.x_train = x_train
        self.y_train = y_train
        self.m, self.n = x_train.shape

    def predict(self, x_test):
        """
        Predict multi-target labels for test data.
        Args:
        - x_test: Test data features (2D array).
        Returns:
        - Predicted multi-target labels (2D array).
        """
        m_test = x_test.shape[0]
        y_pred = np.zeros((m_test, self.y_train.shape[1]), dtype=object) 

        for i in range(m_test):
            # Calculate distances to all training points
            distances = np.array([self._calculate_distance(x_test[i], x) for x in self.x_train])

            # Find the indices of the k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]

            # Get the labels of the k nearest neighbors
            k_nearest_labels = self.y_train[k_indices]

            # Determine the most common label for each target
            for j in range(self.y_train.shape[1]):
                y_pred[i, j] = Counter(k_nearest_labels[:, j]).most_common(1)[0][0]

        return y_pred

    def _calculate_distance(self, x1, x2):
        """
        Calculate the distance between two points based on the chosen metric.
        Args:
        - x1, x2: Two data points (1D arrays).
        Returns:
        - Distance (float).
        """
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.distance_metric == 'minkowski':
            return np.sum(np.abs(x1 - x2) ** self.p) ** (1 / self.p)
        else:
            raise ValueError("Unsupported distance metric. Choose 'euclidean', 'manhattan', or 'minkowski'.")

# Driver code 
  
def main() : 
    df = pd.read_csv( "models/diabetes.csv" ) 
    X = df.iloc[:,:-1].values 
    Y = df.iloc[:,-1:].values 
      
    # Splitting dataset into train and test set 
    X_train, X_test, Y_train, Y_test = train_test_split(  
      X, Y, test_size = 1/3, random_state = 0 ) 
      
    # Model training 
    model = sKNNClassifier()
    model.fit( X_train, Y_train ) 
    model1 = KNeighborsClassifier( n_neighbors = 3 ) 
    model1.fit( X_train, Y_train ) 
      
    # Prediction on test set 
    Y_pred = model.predict( X_test ) 
    Y_pred1 = model1.predict( X_test ) 
      
    # measure performance 
    correctly_classified = 0
    correctly_classified1 = 0
    count = 0
      
    for count in range( np.size( Y_pred ) ) : 
        if Y_test[count] == Y_pred[count] : 
            correctly_classified = correctly_classified + 1
        if Y_test[count] == Y_pred1[count] : 
            correctly_classified1 = correctly_classified1 + 1
        count = count + 1
          
    print( "Accuracy on test set by our model       :  ", (  
      correctly_classified / count ) * 100 ) 
    print( "Accuracy on test set by sklearn model   :  ", (  
      correctly_classified1 / count ) * 100 ) 
      
      
if __name__ == "__main__" :  
    main()