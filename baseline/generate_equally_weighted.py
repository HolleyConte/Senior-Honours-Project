"""
This program takes a chain output by Nautilus and cuts it down to a shorter
length. 

The Nautilus algorithm produces a "weighted" chain - i.e. one where not all
the rows are equally important. This program generates a new chain where all
the rows are equally weighted, by randomly sampling from the original chain 
according to the weights.

This main point of that here is to remove the rows with very low weights
that we don't actually care about.

This script needs cosmosis v3.23 to run, which I'm in the process of releasing (08/09/25).
"""
from astropy.table import Table
# we don't actually use this but it registers the cosmosis table format
import cosmosis.table
import sys
import scipy.special
import numpy as np


input_file = sys.argv[1]
output_file = sys.argv[2]


data = Table.read(input_file, format='cosmosis')
log_weight = data['log_weight']
neff = int(data.meta['neff'])
total_weight = scipy.special.logsumexp(log_weight)
p = np.exp(log_weight - total_weight)

index = np.random.choice(len(p), size=neff, replace=True, p=p)

new_data = data[index]
new_data['log_weight'][:]  = 0.0

new_data.write(output_file, format='cosmosis', overwrite=True)