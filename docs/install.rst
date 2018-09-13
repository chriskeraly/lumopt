Installation and First Optimization
===================================

Install
-------

LumOpt runs on python 2.7, with Lumerical xx or better.

Choose your install directory and run

.. code-block:: bash

    git clone https://github.com/chriskeraly/LumOpt.git
    python setup.py -develop

You will need to add the Lumerical API `lumapi` to your Python path.


Running a prebuilt optimization: a 2D Silicon Photonics Waveguide Y-branch
--------------------------------------------------------------------------

My favorite way of running optimizations is from a jupyter notebook, that way, you can inspect the results in detail after
the optimization, keep a record of the results, or debug the optimization if need be.

In that case just copy the contents of `examples/splitter/splitter_opt.py` into a notebook and run it.

From the terminal:

.. code-block:: bash

    cd examples/splitter
    python splitter_opt_2D.py

Or run the file from your favorite IDE.

If everything is installed correctly, you should see Lumerical windows open, and eventually you should see:

.. raw:: html

    <video width="700" height="400" controls autoplay src="_static/splitter_vid.mp4"></video>
