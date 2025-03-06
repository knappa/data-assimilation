# coding: utf-8
import itertools

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import wasserstein_distance_nd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

colors = ["#004488", "#ddaa33", "#bb5566"]
markers = [".", "1", "+"]

with h5py.File("virt_pop.hdf5", "r") as h5file:
    data = np.log(1e-3 + h5file["virt_pop"][()])


pca = PCA(n_components=2)
pca.fit(data.reshape(-1, 2017 * 41))

t_data = pca.transform(data.reshape(-1, 2017 * 41))

# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=4).fit(data.reshape(-1,2017*41))

gm = GaussianMixture(
    n_components=3,
    means_init=np.array(
        [
            [-198.9468505, -20.54396213],
            [769.86727622, -219.01384513],
            [1074.40147663, 1240.15156448],
        ]
    ),
)
labels = gm.fit_predict(t_data)

fig = plt.figure()
ax = fig.add_subplot()  # projection='3d')

for c_idx in range(3):
    ax.scatter(*t_data[labels == c_idx, :].T, marker=".")
plt.show()

# gaussian-fitted WD in 2d
for i, j in itertools.combinations(range(3), 2):
    print(f"{i=},{j=}")
    ws_dist = 0
    m1 = gm.means_[i, :]
    m2 = gm.means_[j, :]
    C1 = gm.covariances_[i, :, :]
    C2 = gm.covariances_[j, :, :]
    print(
        np.sqrt(
            (m1 - m2) @ (m1 - m2).T
            + np.trace(C1 + C2 - 2 * sqrtm(sqrtm(C2) @ C1 @ sqrtm(C2)))
        )
    )


# discrete WS-dist in 2017*41-d
for i, j in itertools.combinations(range(3), 2):
    print(f"{i=},{j=}")
    print(
        wasserstein_distance_nd(
            data[labels == i].reshape(-1, 2017 * 41),
            data[labels == j].reshape(-1, 2017 * 41),
        )
    )

# discrete WS-dist in 2d
for i, j in itertools.combinations(range(3), 2):
    print(f"{i=},{j=}")
    print(
        wasserstein_distance_nd(
            t_data[labels == i, :],
            t_data[labels == j, :],
        )
    )


# from sklearn.covariance import LedoitWolf
# dists = []
# for idx in range(3):
#    dists.append(LedoitWolf().fit(data[labels == idx].reshape(-1, 2017 * 41)))

bigpca = PCA(n_components=data.shape[0])
bigpca.fit(data.reshape(-1, 2017 * 41))


mus = []
covs = []
for idx in range(3):
    mus.append(
        np.mean(bigpca.transform(data[labels == idx].reshape(-1, 2017 * 41)), axis=0)
    )
    covs.append(
        np.cov(
            bigpca.transform(data[labels == idx].reshape(-1, 2017 * 41)), rowvar=False
        )
    )

# gaussian-fitted WD in 2017*41-d
for i, j in itertools.combinations(range(3), 2):
    print(f"{i=},{j=}")
    m1 = mus[i]
    m2 = mus[j]
    C1 = covs[i]
    C2 = covs[j]
    print(
        np.sqrt(
            (m1 - m2) @ (m1 - m2).T
            + np.trace(C1 + C2 - 2 * sqrtm(sqrtm(C2) @ C1 @ sqrtm(C2)))
        )
    )


# time-parameterized distances
mus = np.zeros((3, 2017, 41))
covs = np.zeros((3, 2017, 41, 41))
dists = np.zeros((3, 2017))
for t_idx in range(2017):
    for l_idx in range(3):
        mus[l_idx, t_idx, :] = np.mean(data[labels == l_idx, t_idx, :], axis=0)
        covs[l_idx, t_idx, :, :] = np.cov(data[labels == l_idx, t_idx, :], rowvar=False)

    for k, (i, j) in enumerate(itertools.combinations(range(3), 2)):
        m1 = mus[i, t_idx, :]
        m2 = mus[j, t_idx, :]
        C1 = covs[i, t_idx, :, :]
        C2 = covs[j, t_idx, :, :]

        dists[k, t_idx] = np.sqrt(
            (m1 - m2) @ (m1 - m2).T
            + np.trace(C1 + C2 - 2 * sqrtm(sqrtm(C2) @ C1 @ sqrtm(C2)))
        )


fig = plt.figure()
ax = fig.add_subplot()
for k, (i, j) in enumerate(itertools.combinations(range(3), 2)):
    ax.plot(dists[k, :], label=f"Distance groups {i} and {j}")
ax.set_title("Wasserstein distance")
fig.legend()
plt.show()


################################################################################


param_pca = PCA(n_components=3)
param_pca.fit(data[:, :, 22:].reshape((data.shape[0], -1)))

fig = plt.figure()
ax = fig.add_subplot()
t_param_data = param_pca.transform(data[:, :, 22:].reshape((data.shape[0], -1)))

from sklearn.svm import LinearSVC

lsvc = LinearSVC(dual="auto", class_weight="balanced")
lsvc.fit(t_param_data[labels != 2, 1:], labels[labels != 2])

for c_idx, mkr in enumerate(markers):
    ax.scatter(
        *t_param_data[labels == c_idx, 1:].T,
        marker=mkr,
        s=16,
        c=colors[c_idx],
        label=f"Cluster {c_idx}",
    )

from sklearn.inspection import DecisionBoundaryDisplay

DecisionBoundaryDisplay.from_estimator(
    lsvc,
    t_param_data[:, 1:],
    ax=ax,
    grid_resolution=50,
    plot_method="contour",
    colors="k",
    levels=[-1, 0, 1],
    alpha=0.5,
    linestyles=["--", "-", "--"],
)
ax.set_title("Clustering of phenotypes in reduced parameter space")
ax.set_axis_off()
fig.tight_layout()
fig.legend(loc="lower right")
plt.savefig("clustering-params-low-dim.pdf")
plt.show()
