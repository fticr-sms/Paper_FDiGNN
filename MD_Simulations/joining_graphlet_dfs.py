# standard library imports
# import os
import pickle
import json
import sys
# from fractions import Fraction

# scientific python imports
import pandas as pd
import numpy as np
import seaborn as sns

graphlets = {}
for i in range(int(sys.argv[-2])):
    graphlets[i] = pd.read_excel(f'MD_TAG_Results/{sys.argv[-1]}_{i}.xlsx').to_dict()

with open(f'{sys.argv[-1]}.json', 'w') as f:
    json.dump(graphlets, f)