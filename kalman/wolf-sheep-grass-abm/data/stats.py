#!/usr/bin/env python3

import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

categories = ["grass", "sheep", "wolf", "wolfgrass", "wolfsheep", "sheepgrass", "wolfsheepgrass"]

full_surprisal_stats = defaultdict(list)
state_surprisal_stats = defaultdict(list)
param_surprisal_stats = defaultdict(list)

for category in categories:
    for idx in range(101):
        try:
            filename = f'{idx:03}-{category}-meansurprisal.csv'
            with open(filename,'r') as file:
                csvreader = csv.reader(file)
                headers = next(csvreader)
                full,state,param = map(float,next(csvreader))
                full_surprisal_stats[category].append(full)
                state_surprisal_stats[category].append(state)
                param_surprisal_stats[category].append(param)
            print(filename)
        except:
            pass

np_full_surprisal_stats = dict()
np_state_surprisal_stats = dict()
np_param_surprisal_stats = dict()


for category in categories:
    np_full_surprisal_stats[category] = np.array(full_surprisal_stats[category])
    np_state_surprisal_stats[category] = np.array(state_surprisal_stats[category])
    np_param_surprisal_stats[category] = np.array(param_surprisal_stats[category])

del full_surprisal_stats
del state_surprisal_stats
del param_surprisal_stats

for category in categories:
    plt.hist(np_full_surprisal_stats[category][np_full_surprisal_stats[category]<200], label=category)

plt.legend()
plt.show()

