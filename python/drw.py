import numpy as np
import pandas as pd
from scipy.misc import comb, factorial

import os
wd = os.path.dirname(os.path.realpath(__file__))
savepath = os.path.join(wd, 'freqs.csv')

cards = range(1,14)
remaining = [4 for _ in cards]

sums = {i: 0 for i in range(8, 77)}

bincounts = {}


for c1 in cards:
    print "c1 " + str(c1)
    for c2 in cards:
        print "c2 " + str(c2)
        for c3 in cards:
            for c4 in cards:
                for c5 in cards:
                    if len(np.unique([c1,c2,c3,c4,c5])) == 1:
                        continue
                    for c6 in cards:
                        if [c1,c2,c3,c4,c5].count(c6) == 4:
                            continue
                        c = np.array([c1,c2,c3,c4,c5,c6])
                        sums[np.sum(c)] += np.product(factorial(np.bincount(c)))



sumdf = pd.DataFrame({"sum": sums.keys(), "count": sums.values()})

sumdf.to_csv(savepath)

print sumdf