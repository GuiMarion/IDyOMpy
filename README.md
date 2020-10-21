# IDyOM
This project propose a Python implementation for the IDyOM model by Marcus Pearce. This implementation was made by Guilhem Marion, Ph.D. student at Laboratoire des Systèmes Perceptifs, ENS, Paris. This work is still in progress but will be officially released soon.

You can find all documentation on this [website](http://guimarion.github.io/IDyOM).
This project also embed unittests.

## Usage

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
      -m MAX_ORDER, --max_order=MAX_ORDER
                            Maximal order to use (default 20).
