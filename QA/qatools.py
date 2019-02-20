import os as _os
import unittest as _unittest
import xmlrunner

TestCase = _unittest.TestCase

def run(filenames):
    for filename in filenames:
        _os.chdir(_os.path.abspath(_os.path.dirname(filename)))
        filename = _os.path.basename(filename)
        module_name = filename.replace(".py", "")
        suite = _unittest.TestLoader().loadTestsFromName(module_name)
        with open(module_name+".xml", "w") as output:
            xmlrunner.XMLTestRunner(output=output, verbosity=2).run(suite)
