import yaml
import sys
import os

import pandas as pd
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))["split"]

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython3 train_test_split.py data-file\n")
    sys.exit(1)

f_input = sys.argv[1]
f_output_train = os.path.join("data", "stage2", "train.csv")
os.makedirs(os.path.join("data", "stage2"), exist_ok=True)
f_output_test = os.path.join("data", "stage2", "test.csv")
os.makedirs(os.path.join("data", "stage2"), exist_ok=True)

p_split_ratio = params["split_ratio"]

#p_split_ratio = 0.3

data = pd.read_csv(f_input)


x=data.drop(['writing'],axis=1)
y=data['writing']

#x.to_csv(f_output_train, index=False)
#y.to_csv(f_output_test, index=False)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=p_split_ratio)

pd.concat([X_train, y_train], axis=1).to_csv(f_output_train, index=None)
pd.concat([X_test, y_test], axis=1).to_csv(f_output_test, index=None)
