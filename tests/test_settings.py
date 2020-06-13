import os
import sys
import unittest

sys.path.insert(0,os.path.join(os.getcwd().split("tests")[0],"app"))

from settings import load_settings, settings

class SettingsTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass
        
    def test_load(self):
        settings["new_entry"] = 0
        self.assertRaises(Exception,load_settings,"app","settings")
        
if __name__ == '__main__':
    unittest.main()
