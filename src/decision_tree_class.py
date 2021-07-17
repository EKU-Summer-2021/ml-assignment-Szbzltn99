"""
decision tree class
"""
from datetime import datetime

from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.csv_read import csv_read


class DecisionTree:
    """
    model class
    """

    def __init__(self):
        self.dataframe = pd.DataFrame(csv_read('C:/Users/nemtu/PycharmProjects/ml-assignment-Szbzltn99'
                                               '/src/csv_files/avocado.csv'))
        self.inputs = self.dataframe.drop('AveragePrice', axis='columns')
        self.target = self.dataframe['AveragePrice']
        le_date = LabelEncoder()
        le_type = LabelEncoder()
        le_region = LabelEncoder()
        self.inputs['date_n'] = le_date.fit_transform(self.inputs['Date'])
        self.inputs['type_n'] = le_type.fit_transform(self.inputs['type'])
        self.inputs['region_n'] = le_region.fit_transform(self.inputs['region'])
        inputs_n = self.inputs.drop(['Date', 'type', 'region'], axis='columns')
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(inputs_n, self.target,
                                                                                test_size=0.33, random_state=42)

    def train(self):
        """
        training the model
        """
        params = {'min_samples_leaf': [5, 6, 7, 8, 9, 10, 11], 'random_state': [0, 1, 2, 3],
                  'min_impurity_decrease': [0.0, 0.1, 0.2, 0.3], 'splitter': ['random', 'best']}
        grid_search_cv = GridSearchCV(DecisionTreeRegressor(random_state=42), params, verbose=3, cv=3)
        grid_search_cv.fit(self.x_train, self.y_train)
        print(grid_search_cv.score(self.x_test, self.y_test))
        return grid_search_cv.best_estimator_

    def put_csv(self):
        """
        Testing and putting the results into csv
        """
        i = 0
        j = 0
        trained_model = self.train()
        self.y_test = self.y_test.tolist()
        y_pred = trained_model.predict(self.x_test)
        dataframe = pd.DataFrame(np.zeros((len(y_pred), 2)))
        while i < len(y_pred):
            dataframe[0][i] = y_pred[i]
            i += 1
        while j < len(y_pred):
            dataframe[1][j] = self.y_test[j]
            j += 1
        dataframe.to_csv('C:/Users/nemtu/PycharmProjects/ml-assignment-Szbzltn99/'
                         'avocado_average_project/results/decision_tree_model/'
                         '2021_07_15_date/avocado_average.csv')
        plt.figure(dpi=70)
        tree.plot_tree(trained_model, max_depth=2)
        plt.savefig('C:/Users/nemtu/PycharmProjects/ml-assignment-Szbzltn99/avocado_average_project/'
                    'results/decision_tree_model/2021_07_15_date/tree.pdf', format='pdf')
        return y_pred

    def save_results(self):
        """
        Method that creates a directory for every single output, and saves it in a csv file with its plot.
        """
        directory = 'Results'
        parent_dir = 'C:/Users/nemtu/PycharmProjects/ml-assignment-Szbzltn99'
        path = os.path.join(parent_dir, directory)
        isdir = os.path.isdir(path)
        if not isdir:
            os.mkdir(path)
        directory_lr = 'LR'
        parent_dir = 'C:/Users/nemtu/PycharmProjects/ml-assignment-Szbzltn99/Results'
        path = os.path.join(parent_dir, directory_lr)
        isdir = os.path.isdir(path)
        if not isdir:
            os.mkdir(path)
        file_location = os.path.join(path,
                                     datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        parent_dir = 'C:/Users/nemtu/PycharmProjects/ml-assignment-Szbzltn99/Results'
        path = os.path.join(parent_dir, directory)
        isdir = os.path.isdir(file_location)
        os.mkdir(file_location)
        self.__save_output_to_csv(file_location)
        self.__save_output_to_plot(file_location)

    def __save_output_to_csv(self, file_location):
        """
            Private method that saves the output into a csv file.
        """
        file_name = file_location + "/" + 'result'
        params = self.search.cv_results_
        params = pd.DataFrame(params)
        params.to_csv(file_name)

    def __save_output_to_plot(self, x_test, y_test):
        """
            Private method that saves the output plot into the csv file.
        """
        plt.plot(x_test, y_test, color='k', label='Decision tree')
        plt.show()


dst = DecisionTree()
dst.put_csv()
