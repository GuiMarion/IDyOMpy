# IDyOM
This project propose a Python implementation for a variant of the IDyOM model by Marcus Pearce. This implementation was made by Guilhem Marion, Ph.D. student at Laboratoire des Systèmes Perceptifs, ENS, Paris. This variant consists in implementing a physiological hypothesis directly into the model: the brain is predicting notes further than the subsequent one(cf. report.pdf for figures and descriptions). This work is temporary paused and will be left here in the meantime.


You can find all documentation on this [website](http://guimarion.github.io/IDyOM).
This project also embed unittests.

## Usage

    Usage: usage App.py [options]

    Options:
      -h, --help            show this help message and exit
      -a AJUMP, --ajump=AJUMP
                            plot comparison with the jump
      -t TRAIN_FOLDER, --train=TRAIN_FOLDER
                            Train the model with the passed folder
      -j JUMP, --jump=JUMP  Use JUMP model as LTM is 1 is passed
      -l TRIAL_FOLDER, --likelihood=TRIAL_FOLDER
                            Compute likelihoods over the passed folder
      -z ZERO_PADDING, --zero_padding=ZERO_PADDING
                            Specify if you want to use zero padding in the
                            surprise output (1 by default)
      -p LISP, --lisp=LISP  plot comparison with the lisp version
      -i FOLDERTRAIN, --in=FOLDERTRAIN
                            Training folder to use
