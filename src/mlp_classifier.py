import os
import pandas as pd
import numpy as np
from datetime import datetime
from src.csv_read import csv_read
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV



class MlPC:
    def __init__(self):
        path = '/winequality-red.csv'
        self.dataframe = pd.DataFrame(csv_read(os.getcwd() + path))
        self.inputs = self.dataframe.drop('quality', axis='columns')
        self.target = self.dataframe['quality']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.inputs, self.target,
                                                                                test_size=0.2, random_state=1)

    def scale(self):
        """
        scaling the data
        """
        sc_x = StandardScaler()
        self.x_train = sc_x.fit_transform(self.x_train)
        self.x_test = sc_x.transform(self.x_test)
        return self.x_train, self.x_test

    def train(self, params):
        self.x_train, self.x_test = self.scale()
        grid_search_cv = GridSearchCV(MLPClassifier(), params, verbose=3, cv=3)
        grid_search_cv.fit(self.x_train, self.y_train)
        print(grid_search_cv.score(self.x_test, self.y_test))
        return grid_search_cv.best_estimator_

    def pred(self, params):
        """
        predicting with the model
        """
        model = self.train(params)
        y_pred = np.array(model.predict(self.x_test))
        i = 0
        j = 0
        self.y_test = self.y_test.tolist()
        dataframe = pd.DataFrame(np.zeros((len(y_pred), 2)))
        while i < len(y_pred):
            dataframe[0][i] = y_pred[i]
            i += 1
        while j < len(y_pred):
            dataframe[1][j] = self.y_test[j]
            j += 1
        path = self.save_results()
        wine = '/wine_quality2.csv'
        dataframe.to_csv(path + wine)
        return y_pred

    @staticmethod
    def save_results():
        """
        Method that creates a directory for every single output, and saves it in a csv file with its plot.
        """
        directory = 'Results'
        parent_dir = os.getcwd()
        path = os.path.join(parent_dir, directory)
        isdir = os.path.isdir(path)
        if not isdir:
            os.mkdir(path)
        directory_lr = 'MLPR'
        path = os.path.join(path, directory_lr)
        isdir = os.path.isdir(path)
        if not isdir:
            os.mkdir(path)
        file_location = os.path.join(path,
                                     datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.mkdir(file_location)
        return file_location


mlp = MlPC()
params = {'hidden_layer_sizes': [3200],
          'activation': ['relu'],
          'max_iter': [1000], 'learning_rate': ['constant'],

          }
mlp.pred(params)
