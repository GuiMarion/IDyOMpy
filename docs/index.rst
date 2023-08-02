.. IDyOM documentation master file, created by
   sphinx-quickstart on Mon Mar  4 14:00:21 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*********************************
Welcome to the IDyOMpy documentation!
*********************************
.. toctree::
   :hidden:
   :maxdepth: 2

   self


This project propose a Python implementation for the IDyOM model made by `Marcus Pearce <https://code.soundsoftware.ac.uk/projects/idyom-project>`_
. 

This code implements an automatic documentation as well as unit tests for every functions. This website hosts all the technical information as well as the documentation of the program. If you use this code, please cite the related paper.


Get Started
***********

You can install this project using pip:
====================================

  python3 -m pip install idyompy

We recommand to install it in a virtual environement for stability.


You can also get the project from GitHub:
====================================

	git clone https://github.com/GuiMarion/IDyOM.git

And install the dependencies:
=====================

  pip install -r requirements.txt

Then use it!
================

Usage: usage App.py [options]

Options:
  -h, --help            show this help message and exit
  -a TESTS, --test=TESTS
                        1 if you want to launch unittests
  -t TRAIN_FOLDER, --train=TRAIN_FOLDER
                        Train the model with the passed folder
  -s TRIAL_FOLDER, --surprise=TRIAL_FOLDER
                        Compute surprise over the passed folder. We use -t
                        argument to train, if none are privided, we use the
                        passed folder to cross-train.
  -n TRIAL_FOLDER_SILENT, --silentNotes=TRIAL_FOLDER_SILENT
                        Compute silent notes probabilities over the passed
                        folder. We use -t argument to train, if none are
                        provided, we use the passed folder to cross-train.
  -d THRESHOLD_MISSING_NOTES, --threshold_missing_notes=THRESHOLD_MISSING_NOTES
                        Define the threshold for choosing the missing notes
                        (0.3 by default)
  -z ZERO_PADDING, --zero_padding=ZERO_PADDING
                        Specify if you want to use zero padding in the
                        surprise output, enable time representation (default
                        0)
  -p LISP, --lisp=LISP  plot comparison with the lisp version
  -b SHORT_TERM_ONLY, --short_term=SHORT_TERM_ONLY
                        Only use short term model (default 0)
  -c CROSS_EVAL, --cross_eval=CROSS_EVAL
                        Compute likelihoods by pieces over the passed dataset
                        using k-fold cross-eval.
  -l LONG_TERM_ONLY, --long_term=LONG_TERM_ONLY
                        Only use long term model (default 0)
  -k K_FOLD, --k_fold=K_FOLD
                        Specify the k-fold for all cross-eval, you can use -1
                        for leave-one-out (default 5).
  -q QUANTIZATION, --quantization=QUANTIZATION
                        Rythmic quantization to use (default 24).
  -v VIEWPOINTS, --viewPoints=VIEWPOINTS
                        Viewpoints to use (pitch, length or both), default
                        both
  -m MAX_ORDER, --max_order=MAX_ORDER
                        Maximal order to use (default 20).


.. toctree::
    :hidden:
    :glob:

    *