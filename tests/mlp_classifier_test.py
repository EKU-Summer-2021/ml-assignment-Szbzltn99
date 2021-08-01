import numpy
import unittest
from src.mlp_classifier import MlPC


class CsvReadTest(unittest.TestCase):
    def test_csv_read_test(self):
        mlpc = MlPC()
        num = 0
        params = {'hidden_layer_sizes': [100],
                  'activation': ['relu'],
                  'max_iter': [2000], 'learning_rate': ['constant'],
                  'alpha': [0.0002, 0.0001],
                  'beta_1': [0.9, 0.99]
                  }
        y_pred = mlpc.pred(params)
        y_actual = mlpc.y_test
        istrue = numpy.isclose(y_pred, y_actual, rtol=0.1)
        for x in istrue:
            if x:
                num = num + 1
        if num/len(istrue) > 0.5:
            actual = True
        expected = True
        self.assertEqual(expected, actual)