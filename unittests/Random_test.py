import unittest
import random
import time

class RandomTest(unittest.TestCase):

	def test_WRONG(self):
		self.assertIn("p", [1,2,3])

	def test_true(self):
		self.assertIn("p", ["p"])

unittest.main()