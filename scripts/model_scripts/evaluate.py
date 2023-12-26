import os
import sys
import pickle
import json

import pandas as pd

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython evaluate.py data-file model\n")
    sys.exit(1)

data = pd.read_csv(sys.argv[1])
X=data.drop(['writing'],axis=1)
y=data['writing']

with open(sys.argv[2], "rb") as fd:
    clf = pickle.load(fd)

score = clf.score(X, y)

prc_file = os.path.join("evaluate", "score.json")
os.makedirs(os.path.join("evaluate"), exist_ok=True)

with open(prc_file, "w") as fd:
    json.dump({"score": score}, fd)
