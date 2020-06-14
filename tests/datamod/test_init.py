import os
import sys
import unittest

import numpy as np

sys.path.insert(0,os.path.join(os.getcwd().split("tests")[0],"app"))

from datamod import normalize, scale

class DataTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass
    
    def test_normalize(self):
        test_list = np.array([-2, 4, 1, 0, 6]).reshape(-1,1)
        data, ranges = normalize(test_list)
        self.assertEqual(np.min(data), 0.0)
        self.assertEqual(np.max(data), 1.0)
        self.assertEqual(ranges[0][0], -2)
        self.assertEqual(ranges[0][1], 6)
    
    def test_scale(self):
        test_data = np.array([[0.],[0.75],[0.375],[0.25 ],[1.]])
        test_ranges = np.array([[-2,6]])
        test_list = scale(test_data,test_ranges)
        expected = np.array([[-2.],[4.],[1.],[0.],[6.]])
        data, ranges = normalize(test_list)
        numpy_test = np.testing.assert_array_equal(test_list,expected)
        self.assertIs(numpy_test,None)
        
if __name__ == '__main__':
    unittest.main()
