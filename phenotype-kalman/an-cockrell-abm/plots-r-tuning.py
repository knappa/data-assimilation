import os

import h5py
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerBase

from transform import transform_kf_to_intrinsic

fig, axs = plt.subplots(1, 1, figsize=(6.5, 5), sharex=True)

finals = []
for p_idx, prefix in enumerate(["10.00", "01.00", "00.10", "00.01"]):
    files = [
        f
        for f in os.listdir(".")
        if os.path.isfile(f) and f.startswith("cyt-" + prefix) and f.endswith(".hdf5")
    ]
    surp_final = []
    for file in files:
        with h5py.File(file, "r") as h5file:
            if np.all(h5file["surprisal_full"][-1, :] < 100_000):
                surp_final.append(h5file["surprisal_full"][-1, :])
            else:
                print(files.index(file))
    surp_final = np.array(surp_final)

    finals.append(
        axs.plot(
            np.median(surp_final, axis=0)[:-1],
            label="final " + prefix,
            color=mpl.colormaps["tab10"](p_idx),
        )[0]
    )

axs.set_ylabel("surprisal")
axs.set_xlabel("time")
# axs.title.set_text("Measuring Cytokines")


# noinspection PyTypeChecker
class AnyObjectHandler(HandlerBase):
    def create_artists(
        self, legend, orig_handle, x0, y0, width, height, fontsize, trans
    ):
        size = len(orig_handle)
        ls = []
        # noinspection PyShadowingNames
        for idx, handle in enumerate(orig_handle):
            h = (size - idx) / (size + 1) * height
            ls.append(
                plt.Line2D(
                    [x0, y0 + width],
                    [h, h],
                    linestyle=handle.get_linestyle(),
                    color=handle.get_color(),
                )
            )

        return ls


axs.legend(
    finals,
    [*map(lambda x: x.get_label(), finals)],
    loc="upper left",
    # handler_map={tuple: AnyObjectHandler()},
)

fig.suptitle("Surprisal after measurement")
fig.tight_layout()
fig.savefig("an-cockrell-r-measurement-uncertainty.pdf")
plt.close(fig)
