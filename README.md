# IDyOMpy
IDyOMpy is a Python re-implementation of the IDyOM model made by Marcus Pearce. This implementation was made by Guilhem Marion, a Ph.D. student at Laboratoire des Systèmes Perceptifs, ENS, Paris. You can find the documentation on this [website](http://guimarion.github.io/IDyOMpy). Please cite the related paper if you use this work.

**IDyOMpy article is out on Journal of Neuroscience Methods!** Go check it out [here](https://doi.org/10.1016/j.jneumeth.2024.110347) and pdf [here](https://guimarion.github.io/docs/Marion2024.pdf).

# Get Started
    
## Install conda if you don't have it (we highly recommend)

visit: https://conda.io/projects/conda/en/latest/user-guide/install/index.html

## Create a new environment

    conda create -n idyompyenv
## Activate environment (you will have to activate it every time you need it and deactivate when you don't need it):

    conda activate idyompyenv
    
## Get the project from GitHub:

    git clone https://github.com/GuiMarion/IDyOM.git

## Enter the project

    cd IDyOM

## Install the dependencies:

    pip install -r requirements.txt

## Then use it!

### To run a cross-validation with 5-folds:
    python3 App.py -c yourFolder/ -k 5
### To train/test:
    python3 App.py -t trainingFolder/ -s testingFolder/
### To replicate figures from Music of Silence Part II:
    python3 App.py -t trainingFolder/ -d testingFolder/
### To change the maximal order to 10:
    python3 App.py -t trainingFolder/ -s testingFolder/ -m 10
### To only use the pitch information:
    python3 App.py -t trainingFolder/ -s testingFolder/ -v pitch
### To compute genuine entropies and not use the approximation (it takes 5 times longer):
    python3 App.py -t trainingFolder/ -s testingFolder/ -g 1
### To compute the training monitoring (c.f. method paper):
    python3 App.py -i preTrainingFolder/ -e trainingFolder/
### To only use the long-term model:
    python3 App.py -t trainingFolder/ -s testingFolder/ -l 1
### To only use the short-term model:
    python3 App.py -t trainingFolder/ -s testingFolder/ -b 1

Of course, all those parameters can be mixed to reach the behavior you are looking for.

## Usage: usage App.py [options]

### Options:
    -h, --help 	show this help message and exit
    -a TESTS, --test=TESTS
     	1 if you want to launch unittests
    -t TRAIN_FOLDER, --train=TRAIN_FOLDER
     	Train the model with the passed folder
    -s TRIAL_FOLDER, --surprise=TRIAL_FOLDER
     	Compute surprise over the passed folder. We use -t argument to train, if none are privided, we use the passed folder to cross-train.
    -n TRIAL_FOLDER_SILENT, --silentNotes=TRIAL_FOLDER_SILENT
     	Compute silent notes probabilities over the passed folder. We use -t argument to train, if none are provided, we use the passed folder to cross-train.
    -d THRESHOLD_MISSING_NOTES, --threshold_missing_notes=THRESHOLD_MISSING_NOTES
     	Define the threshold for choosing the missing notes (0.2 by default)
    -z ZERO_PADDING, --zero_padding=ZERO_PADDING
     	Specify if you want to use zero padding in the surprise output, enable time representation (default 0)
    -b SHORT_TERM_ONLY, --short_term=SHORT_TERM_ONLY
     	Only use short term model (default 0)
    -c CROSS_EVAL, --cross_eval=CROSS_EVAL
     	Compute likelihoods by pieces over the passed dataset using k-fold cross-eval.
    -l LONG_TERM_ONLY, --long_term=LONG_TERM_ONLY
     	Only use long term model (default 0)
    -k K_FOLD, --k_fold=K_FOLD
     	Specify the k-fold for all cross-eval, you can use -1 for leave-one-out (default 5).
    -q QUANTIZATION, --quantization=QUANTIZATION
     	Rythmic quantization to use (default 24).
    -v VIEWPOINTS, --viewPoints=VIEWPOINTS
     	Viewpoints to use: pitch, length, interval, and velocity, separate them with comas, default pitch,length.
    -m MAX_ORDER, --max_order=MAX_ORDER
     	Maximal order to use (default 20).
    -g GENUINE_ENTROPIES, --genuine_entropies=GENUINE_ENTROPIES
     	Use this parameter to NOT use the entropy approximation. It takes longer (5 times) to compute but generate the genuine entropies, not an approximation (default 0).
    -r FOLDER_DUPLICATES, --check_dataset=FOLDER_DUPLICATES
     	Check whether the passed folder contains duplicates.
    -e TRAIN_TEST_FOLDER, --evolution=TRAIN_TEST_FOLDER
     	Train and evaluate over training on the passed folder (cross-val).
    -i INTIALIZATION, --init_evolution=INTIALIZATION
     	Folder to initialize the evolution on.
    -p NB_PIECES, --nb_pieces=NB_PIECES
     	Number of pieces to evaluate on during evolution training.
    -o ORIGINAL_PPM, --original_ppm=ORIGINAL_PPM
     	Use original PPM algorithm to calculate likelihoods(default 0).
