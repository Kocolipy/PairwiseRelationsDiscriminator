import numpy as np
import os
import pickle
import pathlib
import random
from skimage.transform import resize

# file_dir = pathlib.Path("D:\RAVEN\Fold_1")
file_dir = pathlib.Path(os.getcwd()) / "data"

file = "All"
print("Generating for", file)
out_dir = file_dir / file
source_dir = out_dir / "train"

ncd_dir = out_dir / "ncd"
ncd_dir.mkdir(parents=True, exist_ok=True)

files = os.listdir(str(source_dir))

print("Generating NCD samples")
for i, x in enumerate(files):
    if (i+1) % 4200 == 0:
        print(i, "files generated")
    random.shuffle(files)
    random_file = files[0]
    with open(str(source_dir / x), "rb") as f:
        s1 = pickle.load(f)
    with open(str(source_dir / random_file), "rb") as f:
        s2 = pickle.load(f)

    panels = []
    # random_int = random.sample(range(8), 3)
    for j in range(s1.answerPanels.shape[0]):
        # panel = s2.answerPanels[j] if j in random_int else s1.answerPanels[j]
        panel = s1.answerPanels[j] if random.random() > 0.5 else s2.answerPanels[j]
        panels.append(panel)

    matrix = (s1.questionPanels, np.stack(panels))

    with open(str(ncd_dir / str(i)), "wb+") as f:
        pickle.dump(matrix, f)

