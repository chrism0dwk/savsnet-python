# Data extraction tests

import unittest
import os
import pandas as pd

from savsnet.prev_ts_GP import extractData

class TestDataExtraction(unittest.TestCase):

    def setUp(self):
        test_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_path = os.path.join(test_dir, "../data/savsnet_2019_08_06.csv")

    def test_extract(self):
        data_csv = pd.read_csv(self.data_path)
        data = extractData(data_csv, 'cat', 'gastroenteric')
        print(data)