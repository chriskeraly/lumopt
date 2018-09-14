# Continuous adjoint optimization wrapper for Lumerical

## Introduction

This is a continuous adjoint opimtization wrapper for Lumerical, using Python as the main user interface. It is released under an MIT license.

If you use this tool in any published work, please cite https://www.osapublishing.org/oe/abstract.cfm?uri=oe-21-18-21693 and give a link to this repo. Thanks!

## Tutorials, Examples, and Documentation

It is all here: https://lumopt.readthedocs.io/en/latest/

## Install

Make sure you have the latest version of Lumerical installed, and that lumapi (the python api) works

```bash
cd your/install/folder/
git clone https://github.com/chriskeraly/LumOpt.git
python setup.py develop
```

I would strongly recommend using jupyter notebooks to run optimizations.

## First optimization

If you are not using jupyter notebooks:

```bash
cd your/install/folder/examples/Ysplitter
python splitter_opt_2D.py
```

Otherwise copy `your/install/folder/examples/Ysplitter/splitter_opt_2D.py` into a notebook
