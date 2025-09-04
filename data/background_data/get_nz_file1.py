import numpy as np
import matplotlib.pyplot as plt

data = np.load("lsst_source_bins_year_1.npy", allow_pickle=True).item()
z = data['redshift_range']
bins = data['bins']
bins = np.vstack([bins[i] for i in range(5)])
out = np.vstack([z, bins])

# for i in range(1, 6):
#     plt.plot(out[0], out[i], label=f'bin {i-1}')
# plt.show()

np.savetxt("lsst_source_bins_year_1.txt", out.T, header="z bin0 bin1 bin2 bin3 bin4")