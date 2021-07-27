from src.knn_model import KNN
import numpy
import unittest


class CsvReadTest(unittest.TestCase):
    def test_csv_read_test(self):
        knn = KNN()
        num = 0
        params = {'n_neighbors': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]}
        y_pred = knn.pred(params)
        y_actual = knn.y_test
        istrue = numpy.isclose(y_pred, y_actual, rtol=0.1)
        for x in istrue:
            if x:
                num = num + 1
        if num/len(istrue) > 0.5:
            actual = True
        expected = True
        self.assertEqual(expected, actual)
