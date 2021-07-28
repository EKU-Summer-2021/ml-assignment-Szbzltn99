import os
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from src.csv_read import csv_read


class KNN:
    """
    KNN class
    """
    def __init__(self):
        path = '/winequality-red.csv'
        self.dataframe = pd.DataFrame(csv_read(os.getcwd() + path))
        self.inputs = self.dataframe.drop('quality', axis='columns')
        self.target = self.dataframe['quality']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.inputs, self.target,
                                                                                test_size=0.33, random_state=42)

    def plot_correlation(self, path):
        """
        plot config
        """
        # init figure size
        rcParams['figure.figsize'] = 15, 20
        fig = plt.figure()
        sns.heatmap(self.dataframe.corr(), annot=True, fmt=".2f")
        plt.show()
        fig.savefig(path)

    def scale(self):
        """
        scaling the data
        """
        sc_x = StandardScaler()
        self.x_train = sc_x.fit_transform(self.x_train)
        self.x_test = sc_x.transform(self.x_test)
        return self.x_train, self.x_test

    def train(self, params):
        """
        training the model
        """
        self.x_train, self.x_test = self.scale()
        grid_search_cv = GridSearchCV(KNeighborsClassifier(), params, verbose=3, cv=3)
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
        wine = '/wine_quality.csv'
        dataframe.to_csv(path + wine)
        plot = '/plot.png'
        self.plot_correlation(path + plot)
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
        directory_lr = 'KNN'
        path = os.path.join(path, directory_lr)
        isdir = os.path.isdir(path)
        if not isdir:
            os.mkdir(path)
        file_location = os.path.join(path,
                                     datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.mkdir(file_location)
        return file_location
