import unittest
from src.csv_read import csv_read
import pandas
import os


class CsvReadTest(unittest.TestCase):
    def test_csv_read_test(self):
        avocado = '/avocado.csv'
        expected = True
        actual = isinstance(csv_read(os.getcwd() + avocado), pandas.DataFrame)
        self.assertEqual(expected, actual)
