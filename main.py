from src.decision_tree_class import DecisionTree
from src.knn_model import KNN

if __name__ == '__main__':
    dst = DecisionTree()
    dst.put_csv()
    clf = KNN()
    clf.pred()
