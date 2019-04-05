.. IDyOM documentation master file, created by
   sphinx-quickstart on Mon Mar  4 14:00:21 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*********************************
Welcome to IDyOM's documentation!
*********************************
.. toctree::
   :hidden:
   :maxdepth: 2

   self


This project propose a Python implementation for the IDyOM model made by `Marcus Pearce <https://code.soundsoftware.ac.uk/projects/idyom-project>`_
. 

This code implements an automatic documentation as well as unit tests for every functions. The code is currently under developpement, however, this website will be updated regularly.


Get Started
***********

You can get the project from GitHub:
====================================

	git clone https://github.com/GuiMarion/IDyOM.git

Install dependencies:
=====================

  pip install -r requirements.txt

And then use it!
================

Usage: usage App.py [options]

Options:
  -h, --help            show this help message and exit
  -t TESTS, --test=TESTS
                        1 if you want to launch unittests
  -o FOLDER, --opti=FOLDER
                        launch optimisation of hyper parameters on the passed
                        dataset
  -c CHECK, --check=CHECK
                        check the passed dataset
  -g GENERATE, --generate=GENERATE
                        generate piece of the passed length
  -s SURPRISE, --surprise=SURPRISE
                        return the surprise over a given dataset


.. toctree::
    :hidden:
    :glob:

    *