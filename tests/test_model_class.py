import os
import sys
import unittest

sys.path.insert(0,os.path.join(os.getcwd().split("tests")[0],"app"))

from model_class import Model
from settings import load_settings, settings

class ModelTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        temp = settings["root"]
        settings.clear() 
        settings["root"] = temp
        
    def test_init(self):
        settings.update(load_settings("tests","minimal_with"))
        settings["optimization"]["optimize"] = True
        settings["optimization"]["constrained"] = False
        model = Model()

        self.assertEqual(model.n_const, 0)

        self.assertFalse(model.trained)
        self.assertEqual(model.no_samples, 0)
        self.assertEqual(model.sampling_iterations, 0)


        self.assertFalse(model.optimization_converged)

    def test_init_raise(self):
        settings.update(load_settings("tests","minimal"))
        settings["data"]["evaluator"] = "nonsense"

        self.assertRaises(ValueError, Model)
        
if __name__ == '__main__':
    unittest.main()
