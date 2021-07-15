import unittest
from src.csv_read import csv_read
import pandas


class CsvReadTest(unittest.TestCase):
    def test_csv_read_test(self):

        expected = True
        actual = isinstance(csv_read('C:/Users/nemtu/PycharmProjects/ml-assignment-Szbzltn99/'
                                     'src/csv_files/avocado.csv'), pandas.DataFrame)
        self.assertEqual(expected, actual)
