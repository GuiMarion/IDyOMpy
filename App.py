"""
Enter point of the program.
"""
import idyom

from optparse import OptionParser
import unittest

def main():
	"""
	Call this method to easily use the program.
	"""

	pass

if __name__ == "__main__":

	usage = "usage %prog [options]"
	parser = OptionParser(usage)

	parser.add_option("-t", "--test", type="int",
					  help="1 if you want to launch unittests",
					  dest="tests", default=0)

	options, arguments = parser.parse_args()


	if options.tests == 1:
		loader = unittest.TestLoader()
		start_dir = "unittests"
		suite = loader.discover(start_dir)

		runner = unittest.TextTestRunner()
		runner.run(suite)

