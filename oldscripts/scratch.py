import os
import pathlib
import pickle
from shutil import copyfile
import numpy as np
import random
import utils

# file_dir = pathlib.Path(os.getcwd()) / "data" / "All"
file_dir = pathlib.Path(r"D:\RAVEN\Fold_1") / "All"

test_directory = file_dir / "test"
val_directory = file_dir / "val"
train_directory = file_dir / "train"

dis = 12000

for i in range(5):
    f = pickle.load(open(str(train_directory / str(dis + random.randrange(6000))), "rb"))
    print(f.answer)
    utils.displaySample(np.concatenate([f.questionPanels, f.answerPanels]))

# for file in os.listdir(str(val_directory)):
#     copyfile(str(val_directory/file), str(train_directory/str(int(file) + 42000)))
#
# for file in os.listdir(str(test_directory)):
#     copyfile(str(test_directory/file), str(train_directory/str(int(file) + 56000)))

# size = {"train":6000}
#
# for dir in ["train", "val", "test"]:
#     indir = directory/dir
#     outdir = AllDir/dir
#
#     for file in os.listdir(indir):
#         copyfile(indir/file, outdir/str(int(file) + position*size[dir]))
