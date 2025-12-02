# DATA VARIABLES
DATASET = "GiW-selected"

INP_DIR = "/Users/lukas/dev/master/dependencies/ACE-DNV/GiW/"
DATA_DIR = "/Users/lukas/dev/master/data/data_set_kothari/"
CH2STREAM_MODEL_DIR = "/Users/lukas/dev/master/dependencies/ACE-DNV/weights/2ch2stream/2ch2stream_notredame.t7"  # Directory to the saved weights for 2ch2stream network

EXP = "1-2"  # two digits: first one experiment number according to paper. second, subconditions
# Experiment 1: 1)IndoorWalk 2)BallCatch 3)visualSearch
# Experiment 2: 1)all data, separate GFi and GFo, 2) All data, combined GFi and GFo (similar to what GiW did)
# Experiment 3: Indoorwalk + ball catch with agreement labels

VALID_METHOD = "LOO"  # choose between 'LOO' for leave one out or "TTS" for train/test split
GFiGFo_Comb = False  # considering gaze fixation and gaze following as same class (class 0). this is the defult. For exp 2-2, it changes to True below.
LABELER = 6  # labeler number or 'MV' for majority voting or 'AG' for agreements or "NV" for no validation and just testing saved models
OUT_DIR = INP_DIR + "res/"

ET_FREQ = 300

# REPRESENTAION
VISUALIZE = False  # Set to True if you would like to have Feature Histogram, Confusion matrices, and distribution hist of each epoch


# INPUT DATA INFORMATION
VIDEO_SIZE = [1920, 1080]  # [width, height]
PRE_OPFLOW = False  # set to True if optic is provided and is not needed to be computed


# FRAMEWORK VARIABLES
PATCH_SIZE = 64
LAMBDA = 0.0  # Momentum parameter to regulize patch similarity fluctuation
PATCH_PRIOR_STEPS = (
    3  # Number of previouse frames taking part in regulazing next patch content similarty
)
POSTPROCESS_WIN_SIZE = 6  # Size of window running on extracted events for voting.


if EXP == "1-1":  # single task: Indoor walk
    RECS_LIST = "files_task1_lblr5.txt"
    LABELER = 5
elif EXP == "2-1" or EXP == "2-2":  # all tasks together
    RECS_LIST = "files_all_lblr6.txt"
elif EXP == "1-2":  # single task: Ball catch
    RECS_LIST = "files_task2_lblr6.txt"
elif EXP == "1-3":  # single task: Visual search
    RECS_LIST = "files_tasks3_lblr6.txt"
elif EXP == "3":  # all tasks with agreement labels
    RECS_LIST = "files_ag.txt"
if EXP == "2-2":
    GFiGFo_Comb = True
