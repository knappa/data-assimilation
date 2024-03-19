import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(3, sharex=True, sharey=True)

for p_idx, prefix in enumerate(["g-", "s-", "w-"]):
    for r_idx, r_form in enumerate(["10.00", "01.00", "00.10", "00.01"]):
        files = [
            f
            for f in os.listdir(".")
            if os.path.isfile(f) and f[:7] == prefix + r_form and f[-5:] == ".hdf5"
        ]

        surp_init = []
        surp_final = []

        for file in files:
            with h5py.File(file, "r") as h5file:
                surp_init.append(h5file["surprisal_full"][0, :])
                surp_final.append(h5file["surprisal_full"][-1, :])
        surp_init = np.array(surp_init)
        surp_final = np.array(surp_final)
        print(surp_init.shape)

        axs[p_idx].plot(np.mean(surp_init, axis=0))
        axs[p_idx].plot(np.mean(surp_final, axis=0)[:-1], label="final " + r_form)
    axs[p_idx].legend()
