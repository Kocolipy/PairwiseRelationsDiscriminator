import os
import pathlib
import pickle
import shutil

import sample

cwd = pathlib.Path(os.getcwd())

file_dir = cwd / "RAVEN-10000"

# Create directory to contain processed RPM problems
out_dir = cwd / "data"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "train").mkdir(parents=True, exist_ok=True)
(out_dir / "val").mkdir(parents=True, exist_ok=True)
(out_dir / "test").mkdir(parents=True, exist_ok=True)

for count, config in enumerate(sorted(os.listdir(str(file_dir)))):
    config_dir = file_dir / config
    train = []
    val = []
    test = []
    for file in os.listdir(str(config_dir)):
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

    print("Processing Training Set of", config)
    for i, x in enumerate(train):
        s = sample.Sample(config_dir / "RAVEN_{}_train.npz".format(x), config_dir / "RAVEN_{}_train.xml".format(x))
        pickle.dump(s, open(str(out_dir / "train" / str(count*6000 + i)), 'wb+'))

        if (i + 1) % 600 == 0:
            print((i + 1) / 60, "% completed")

    print("Processing Validation Set of", config)
    for i, x in enumerate(val):
        s = sample.Sample(config_dir / "RAVEN_{}_val.npz".format(x), config_dir / "RAVEN_{}_val.xml".format(x))
        pickle.dump(s, open(str(out_dir / "val" / str(count*2000 + i)), 'wb+'))

        if (i + 1) % 200 == 0:
            print((i + 1) / 20, "% completed")

    print("Processing Test Set of", config)
    for i, x in enumerate(test):
        s = sample.Sample(config_dir / "RAVEN_{}_test.npz".format(x), config_dir / "RAVEN_{}_test.xml".format(x))
        pickle.dump(s, open(str(out_dir / "test" / str(count*2000 + i)), 'wb+'))

        if (i+1) % 200 == 0:
            print((i+1)/20, "% completed")
