from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import sys
import os
import yaml
import pickle

import pandas as pd

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython dt.py data-file model \n")
    sys.exit(1)

f_input = sys.argv[1]
f_output = os.path.join("models", sys.argv[2])
os.makedirs(os.path.join("models"), exist_ok=True)

params = yaml.safe_load(open("params.yaml"))["train"]
n_jobs = params["n_jobs"]
fit_intercept = params["fit_intercept"]

data = pd.read_csv(f_input)
#x = df.iloc[:,[1,2,3]]
#y = df.iloc[:,0]
X=data.drop(['writing'],axis=1)
y=data['writing']


lr= LinearRegression(n_jobs=n_jobs, fit_intercept=fit_intercept)
model=lr.fit(X, y)

with open(f_output, "wb") as fd:
    pickle.dump(model, fd)
