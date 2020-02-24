import os
import pathlib
import pickle
from shutil import copyfile

position = 4
directory = pathlib.Path(r"D:\RAVEN\Fold_1\O-IC")
AllDir = pathlib.Path(r"D:\RAVEN\Fold_1\All")

# size = {"train":6000}
#
# for dir in ["train", "val", "test"]:
#     indir = directory/dir
#     outdir = AllDir/dir
#
#     for file in os.listdir(indir):
#         copyfile(indir/file, outdir/str(int(file) + position*size[dir]))
