# Continuous adjoint optimization wrapper for Lumerical

## Introduction

This is a continuous adjoint opimtization wrapper for Lumerical, using Python as the main user interface. It is released under an MIT license.

## Tutorials, examples, and Documentation

It is all here: link

## Install

Make sure you have Lumerical installed, and that lumapi (the python api) works

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
