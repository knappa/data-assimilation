import matplotlib.pyplot as plt
import numpy as np

X = np.linspace(-10, 10, 100)

T = np.linspace(-1, np.log(9), 50)

center_left = -np.exp(T)
center_right = np.exp(T)

data_left = np.exp(-0.5 * (X - center_left[:, None]) ** 2)
data_right = np.exp(-0.5 * (X - center_right[:, None]) ** 2)

#####

fig = plt.figure(figsize=(8, 8), facecolor='black')
ax = plt.subplot(frameon=False)
# Generate line plots
lines = []
for i in range(len(data_left))[::-1]:
    # Small reduction of the X extents to get a cheap perspective effect
    xscale = 1 - i / 200.
    # Same for linewidth (thicker strokes on bottom)
    lw = 1.5 - i / 100.0

    line, = ax.plot(xscale * X, i + data_left[i] / 2.0 + data_right[i] / 2.0, color="w", lw=lw)
    lines.append(line)

#####

fig = plt.figure(figsize=(8, 8), facecolor='black')

num_frames = len(T)
for i in range(num_frames)[::-1]:
    scale = float(i / num_frames)
    ax = fig.add_axes([i / num_frames / 4.0, i / num_frames, 1 - 2 * i / num_frames / 4.0, 1 / num_frames],
                      facecolor='black', alpha=0.0, clip_on=False, zorder=-i)
    ax.plot(X, data_left[i] / 2.0 + data_right[i] / 2.0, color="w")
    ax.set_ylim(0.0, 1.2)

#####

fig = plt.figure(figsize=(8, 8), facecolor='black')

num_frames = len(T)
for i in range(num_frames)[::-1]:
    scale = float(i / num_frames)
    ax = fig.add_axes([i / num_frames / 4.0, i / num_frames, 1 - 2 * i / num_frames / 4.0, 1 / num_frames],
                      facecolor='black', alpha=0.0, clip_on=False, zorder=-i)
    ax.plot(X, data_left[i] / 2.0 + data_right[i] / 2.0, color="w")
    ax.stackplot(X, data_left[i] / 2.0, data_right[i] / 2.0, colors=['b', 'y'])
    ax.set_ylim(0.0, 1.2)

#####

fig = plt.figure(figsize=(8, 8), facecolor='black')

num_frames = len(T)
for i in range(num_frames)[::-1]:
    scale = float(i / num_frames)
    ax = fig.add_axes((0.05, i / num_frames, 0.9, 1 / num_frames),
                      facecolor='black', alpha=0.0, clip_on=False, zorder=-i)
    ax.plot(X, data_left[i] / 2.0 + data_right[i] / 2.0, color="w")
    ax.stackplot(X, data_left[i] / 2.0, data_right[i] / 2.0, colors=['b', 'y'])
    ax.set_ylim(0.0, 1.2)

#####

X = np.linspace(-10, 10, 100)
T = np.linspace(-1, np.log(9), 50)

center_left = -np.exp(T)
center = np.zeros_like(T)
center_right = np.exp(T)

data_left = np.exp(-0.5 * (X - center_left[:, None]) ** 2)
data_center = np.exp(-0.5 * (X - center[:, None]) ** 2)
data_right = np.exp(-0.5 * (X - center_right[:, None]) ** 2)

fig = plt.figure(figsize=(8, 8), facecolor='black')

num_frames = len(T)
for i in range(num_frames)[::-1]:
    scale = float(i / num_frames)
    ax = fig.add_axes((0.05, i / num_frames, 0.9, 1 / num_frames),
                      facecolor='black', alpha=0.0, clip_on=False, zorder=-i)
    ax.plot(X, data_left[i] / 3.0 + data_center[i] / 3.0 + data_right[i] / 3.0, color="w")
    ax.stackplot(X, data_left[i] / 3.0, data_center[i] / 3.0, data_right[i] / 3.0, colors=['b', 'g', 'y'])
    ax.set_ylim(0.0, 1.2)
