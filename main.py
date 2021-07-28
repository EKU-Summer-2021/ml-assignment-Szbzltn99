from src.knn_model import KNN

if __name__ == '__main__':
    #dst = DecisionTree()
    #dst.put_csv()
    clf = KNN()
    params = {'n_neighbors': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
              'weights': ['uniform', 'distance'],
              'algorithm': ['auto', 'ball_tree', 'kd_tree'], 'p': [1, 2, 3],
              'leaf_size': [10, 15, 20, 25, 30, 35, 40, 45, 50]}
    clf.pred(params)
