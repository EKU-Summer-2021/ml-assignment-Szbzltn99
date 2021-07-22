from src.knn_model import KNN
import numpy
import unittest


class CsvReadTest(unittest.TestCase):
    def test_csv_read_test(self):
        knn = KNN()
        num = 0
        y_pred = knn.pred()
        y_actual = knn.y_test
        istrue = numpy.isclose(y_pred, y_actual, rtol=0.1)
        for x in istrue:
            if x:
                num = num + 1
        if num/len(istrue) > 0.5:
            actual = True
        expected = True
        self.assertEqual(expected, actual)