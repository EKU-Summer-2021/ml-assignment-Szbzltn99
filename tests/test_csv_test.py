import unittest
from src.csv_read import csv_read
import pandas
import os


class CsvReadTest(unittest.TestCase):
    def test_csv_read_test(self):
        wine = '/winequality-red.csv'
        expected = True
        actual = isinstance(csv_read(os.getcwd() + wine), pandas.DataFrame)
        self.assertEqual(expected, actual)
