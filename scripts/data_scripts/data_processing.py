import pandas as pd
import os
import sys
import io

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython3 get_features.py data-file\n")
    sys.exit(1)

f_input = sys.argv[1]

df=pd.read_csv(f_input)

categorical=df.drop(['math score','reading score', 'writing score'], axis=1)
numerical=df[['math score','reading score', 'writing score']]

df1= categorical.apply(lambda x: pd.factorize(x)[0])

data=pd.concat([df1,numerical],axis=1,ignore_index=True)

new_columns_name={0:'gender',1:'race',2:'parent education',3:'lunch',4:'preparetion tests',5:'math',6:'reading',7:'writing'}
data=data.rename(columns=new_columns_name)

os.makedirs(os.path.join("data", "stage1"), exist_ok=True)
f_output = os.path.join("data", "stage1", "exams.csv")
data.to_csv(f_output, index=False)
