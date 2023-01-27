import os

ROOT_DIR = "Complete_dataset"
CTR_DIR = "CT json"
TRAIN_FILE = "train.json"
DEV_FILE = "dev.json"
TEST_FILE = "test.json"

TRAIN_PATH = os.path.join(ROOT_DIR, TRAIN_FILE)
DEV_PATH = os.path.join(ROOT_DIR, DEV_FILE)
TEST_PATH = os.path.join(ROOT_DIR, TEST_FILE)
CTR_DIR_PATH = os.path.join(ROOT_DIR, CTR_DIR)



### MODEL

MODEl_CHECKPOINT = "roberta-base"
TASK = "NLI"
