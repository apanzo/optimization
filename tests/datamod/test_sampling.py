import os
import sys
import unittest

sys.path.insert(0,os.path.join(os.getcwd().split("tests")[0],"app"))

from datamod.sampling import sample

##class TestIt(unittest.TestCase):
##    def test_it(self):
##        """
##        Test that it can sum a list of integers
##        """
##        self.assertEqual(1, 1)

##class TestSampling(unittest.TestCase):
##    def test_sample(self):
##        """
##        Test that it can sum a list of integers
##        """
##        self.assertRaises(NameError, sample, "nonsense", 1, 1)

##for i in sys.path:
##    print(i)

if __name__ == '__main__':
    unittest.main()
