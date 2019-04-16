import os
import sys
import unittest
import xmlrunner

CWD = os.path.dirname(os.path.abspath(__file__))
INVERSE_DESIGN_DIR = os.path.join(CWD, "..") 
sys.path.insert(0, INVERSE_DESIGN_DIR)

TestCase = unittest.TestCase

def run(filenames):
    for filename in filenames:
        os.chdir(os.path.abspath(os.path.dirname(filename)))
        filename = os.path.basename(filename)
        module_name = filename.replace(".py", "")
        suite = unittest.TestLoader().loadTestsFromName(module_name)
        with open(module_name+".xml", "w") as output:
            xmlrunner.XMLTestRunner(output=output, verbosity=2).run(suite)
