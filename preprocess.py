import os
import pathlib
import pickle
import shutil
from tqdm import tqdm

from prd.sample import Sample

cwd = pathlib.Path(os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="RAVEN", choices=["RAVEN", "I-RAVEN"])
parser.add_argument('--out_dir', type=str, default="data")
args = parser.parse_args()

file_dir = cwd / f"{args.dataset}-10000"

# Create directory to contain processed RPM problems
out_dir = cwd / args.out_dir
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
    for i, x in enumerate(tqdm(train)):
        s = Sample(config_dir / f"RAVEN_{x}_train.npz", config_dir / f"RAVEN_{x}_train.xml")
        pickle.dump(s, open(str(out_dir / "train" / str(count*6000 + i)), 'wb+'))

    print("Processing Validation Set of", config)
    for i, x in enumerate(tqdm(val)):
        s = Sample(config_dir / f"RAVEN_{x}_val.npz", config_dir / f"RAVEN_{x}_val.xml")
        pickle.dump(s, open(str(out_dir / "val" / str(count*2000 + i)), 'wb+'))

    print("Processing Test Set of", config)
    for i, x in enumerate(tqdm(test)):
        s = Sample(config_dir / f"RAVEN_{x}_test.npz", config_dir / f"RAVEN_{x}_test.xml")
        pickle.dump(s, open(str(out_dir / "test" / str(count*2000 + i)), 'wb+'))
