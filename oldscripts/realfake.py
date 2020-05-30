import numpy as np
import os
import pickle
import pathlib
import random
import shutil

import dualsets
import sample

# file_dir = pathlib.Path(os.getcwd()) / "data"

# for file in os.listdir(file_dir):
file = "3x3Grid"
print("Generating for", file)
out_dir = file_dir / file
source_dir = out_dir / "train"

# if os.path.exists(out_dir / "real"):
#     shutil.rmtree(out_dir / "real")
# (out_dir / "real").mkdir(parents=True)
#
# if os.path.exists(out_dir / "fake"):
#     shutil.rmtree(out_dir / "fake")
# (out_dir / "fake").mkdir(parents=True)

files = os.listdir(str(source_dir))

print("Generating real dual sets samples")
real_dir = out_dir / "real"
real_dir.mkdir(exist_ok=True)
for i, x in enumerate(files):
    with open(str(source_dir / x), "rb") as f:
        s = pickle.load(f)

    ds = dualsets.DualSets.real(s)

    with open(str(real_dir / str(i)), "wb+") as f:
        pickle.dump(ds, f)


print("Generating fake dual sets samples")
fake_dir = out_dir / "fake"
fake_dir.mkdir(exist_ok=True)
for i, x in enumerate(files):
    # [a, b] = random.choices(files, k=2)
    # random.shuffle(files)
    # [a, b] = files[:2]

    with open(str(source_dir / x), "rb") as f:
        s1 = pickle.load(f)
    # with open(str(source_dir / b), "rb") as f:
    #     s2 = pickle.load(f)

    ds = dualsets.DualSets.fake1(s1)
    with open(str(fake_dir / str(i)), "wb+") as f:
        pickle.dump(ds, f)
