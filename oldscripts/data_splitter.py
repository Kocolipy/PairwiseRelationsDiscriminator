import os
import pathlib
import pickle
import shutil

import sample

dirs = {
         "center_single": "Center",
         "distribute_four": "2x2Grid",
         "distribute_nine": "3x3Grid",
         "in_center_single_out_center_single": "O-IC",
         "in_distribute_four_out_center_single": "O-IG",
         "left_center_single_right_center_single": "L-R",
         "up_center_single_down_center_single": "U-D",
         }


for count, key in enumerate(dirs.keys()):
    file_dir = pathlib.Path("D:\RAVEN\RAVEN-10000") / str(key)
    out_dir = pathlib.Path("D:\RAVEN\Fresh")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "val").mkdir(parents=True, exist_ok=True)
    (out_dir / "test").mkdir(parents=True, exist_ok=True)

    train = []
    val = []
    test = []
    for file in os.listdir(file_dir):
        if "train" in file:
            train.append(file)
        elif "val" in file:
            val.append(file)
        elif "test" in file:
            test.append(file)
        else:
            print("Unrecognised file name in directory:", file)

    train = set(file.split('_')[1] for file in train)
    val = set(file.split('_')[1] for file in val)
    test = set(file.split('_')[1] for file in test)

    print("Processing Training Set of", dirs[key])
    for i, x in enumerate(train):
        if (i+1) % 600 == 0:
            print((i+1)/60, "% completed")
        s = sample.Sample(file_dir / "RAVEN_{}_train.npz".format(x), file_dir / "RAVEN_{}_train.xml".format(x))
        with open(out_dir / "train" / str(count*6000 + i), 'wb+') as f:
            pickle.dump(s, f)

    print("Processing Validation Set of", dirs[key])
    for i, x in enumerate(val):
        if (i+1) % 200 == 0:
            print((i+1)/20, "% completed")
        s = sample.Sample(file_dir / "RAVEN_{}_val.npz".format(x), file_dir / "RAVEN_{}_val.xml".format(x))
        with open(out_dir / "val" / str(count*2000 + i), 'wb+') as f:
            pickle.dump(s, f)

    print("Processing Test Set of", dirs[key])
    for i, x in enumerate(test):
        if (i+1) % 200 == 0:
            print((i+1)/20, "% completed")
        s = sample.Sample(file_dir / "RAVEN_{}_test.npz".format(x), file_dir / "RAVEN_{}_test.xml".format(x))
        with open(out_dir / "test" / str(count*2000 + i), 'wb+') as f:
            pickle.dump(s, f)
