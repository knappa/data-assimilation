import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

num_sims = 2000
num_param_levels = 5
levels = {"0.8": 0, "0.9": 1, "1.0": 2, "1.1": 3, "1.2": 4}

param_names = dict()

hdf_files = [
    f for f in os.listdir(".") if f.endswith(".hdf5") and f.startswith("statistics")
]

data = np.zeros((64, num_param_levels, num_sims), dtype=np.float64)

for filename in hdf_files:
    fields = filename.split("-")
    param_idx = int(fields[2])
    param_name = fields[3]
    param_level = levels[fields[5][:-5]]
    param_names[param_name] = param_idx
    with h5py.File(filename, "r") as file:
        data[param_idx, param_level, :] = np.sort(np.min(file["system_health"], axis=1))

param_idcs = np.zeros(len(param_names), dtype=object)
for name, idx in param_names.items():
    param_idcs[idx] = name

with h5py.File("min_system_health_param_statistics.hdf5", "w") as file:
    file["data"] = data
    file["params"] = param_idcs
params = param_idcs


def fix_title(s: str, *, break_len=14):
    """
    Fix variable name titles.

    :param s: a title with _'s and maybe too long
    :param break_len: where to look for line breaks
    :return: a title without _'s and with \n's in reasonable places
    """
    s = s.split("_")
    lines = []
    line = ""
    for word in s:
        if len(s) + len(line) + 1 > break_len:
            lines.append(line)
            line = word
        else:
            line = line + " " + word
    if len(line) > 0:
        lines.append(line)
    return "\n".join(lines)


# fix for early failure to remove junk parameter (oops)
gap_idx = list(params).index(b"resistance_to_infection")

fig, axs = plt.subplots(9, 7, sharey=True, sharex=True, figsize=(8.5, 11))
for param_idx_t in range(64):
    # fix to skip junk param
    if param_idx_t < gap_idx:
        param_idx = param_idx_t
    elif param_idx_t == gap_idx:
        continue
    else:
        param_idx = param_idx_t - 1

    r, c = divmod(param_idx, 7)
    for level_idx in range(5):
        axs[r, c].ecdf(data[param_idx, level_idx, :])
    axs[r, c].tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )
    if issubclass(params[param_idx], bytes):
        title = bytes(params[param_idx]).decode("utf-8")
    else:
        title = str(params[param_idx])
    axs[r, c].set_title(fix_title(title), fontsize=8, wrap=True)
fig.subplots_adjust(
    top=0.97, bottom=0.01, left=0.035, right=0.995, hspace=0.8, wspace=0.11
)
plt.savefig("cdf-plots-for-ks-test.pdf")
# plt.show()
plt.close()


# KS-test


def ks_statistic(set_a, set_b):
    set_a = np.sort(set_a)
    set_b = np.sort(set_b)
    assert len(set_a) == len(set_b)
    l = len(set_a)
    i, j = 0, 0
    max_gap = 0
    while i < l and j < l:
        if set_a[i] < set_b[j]:
            i += 1
            max_gap = max(max_gap, abs(i - j))
        elif set_a[i] > set_b[j]:
            j += 1
            max_gap = max(max_gap, abs(i - j))
        else:
            i += 1
            j += 1
    return max_gap / l


alpha = 0.05
rejection_level = np.sqrt(-np.log(alpha / 2) / 2) * np.sqrt(
    (num_sims + num_sims) / (num_sims * num_sims)
)

fig, axs = plt.subplots(9, 7, sharey=True, sharex=True, figsize=(8.5, 11))
for param_idx_t in range(64):
    # fix to skip junk param
    if param_idx_t < gap_idx:
        param_idx = param_idx_t
    elif param_idx_t == gap_idx:
        continue
    else:
        param_idx = param_idx_t - 1

    r, c = divmod(param_idx, 7)
    axs[r, c].plot(
        [0.8, 0.9, 1.0, 1.1, 1.2],
        [
            ks_statistic(data[param_idx, 0, :], data[param_idx, 2, :]),
            ks_statistic(data[param_idx, 1, :], data[param_idx, 2, :]),
            0,
            ks_statistic(data[param_idx, 2, :], data[param_idx, 3, :]),
            ks_statistic(data[param_idx, 2, :], data[param_idx, 4, :]),
        ],
        "-o",
    )
    # axs[r, c].plot([0.85, 0.95, 1.05, 1.15], [
    #     ks_statistic(data[param_idx, 0, :], data[param_idx, 1, :]),
    #     ks_statistic(data[param_idx, 1, :], data[param_idx, 2, :]),
    #     ks_statistic(data[param_idx, 2, :], data[param_idx, 3, :]),
    #     ks_statistic(data[param_idx, 3, :], data[param_idx, 4, :])
    # ],'-o')
    axs[r, c].plot([0.8, 1.2], [rejection_level, rejection_level], ls=":", c="k")
    # axs[r, c].tick_params(
    #     axis='x',  # changes apply to the x-axis
    #     which='both',  # both major and minor ticks are affected
    #     bottom=False,  # ticks along the bottom edge are off
    #     top=False,  # ticks along the top edge are off
    #     labelbottom=False)
    title = bytes(params[param_idx]).decode("utf-8")
    axs[r, c].set_title(fix_title(title), fontsize=8, wrap=True)
    axs[r, c].set_xticks([0.8, 0.9, 1.0, 1.1, 1.2])
fig.subplots_adjust(
    top=0.97, bottom=0.01, left=0.035, right=0.995, hspace=0.8, wspace=0.11
)
# plt.savefig('ks-plots.pdf')
plt.show()
# plt.close()


# print names of vars passing the sensitivity test
alpha = 0.01
rejection_level = np.sqrt(-np.log(alpha / 2) / 2) * np.sqrt(
    (num_sims + num_sims) / (num_sims * num_sims)
)
params_pass = []
for param_idx in range(64):
    if (
        max(
            ks_statistic(data[param_idx, 0, :], data[param_idx, 2, :]),
            ks_statistic(data[param_idx, 1, :], data[param_idx, 2, :]),
            ks_statistic(data[param_idx, 2, :], data[param_idx, 3, :]),
            ks_statistic(data[param_idx, 2, :], data[param_idx, 4, :]),
        )
        > rejection_level
    ):
        params_pass.append(params[param_idx])
