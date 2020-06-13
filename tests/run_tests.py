import unittest

verbose = 2

loader = unittest.TestLoader()
testSuite = loader.discover("..",pattern='test_*.py', top_level_dir="..")
testRunner = unittest.TextTestRunner(verbosity=verbose)
testRunner.run(testSuite)

input()
