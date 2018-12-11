#!/usr/bin/env python3
import numpy as np

procs = [1, 2, 4, 16, 64]
times = [
    [0.000131, 0.000228, 0.000257, 0.000440, 0.0351],
    [0.000223, 0.000315, 0.000319, 0.000860, 0.0263],
    [0.000678, 0.000499, 0.000476, 0.000437, 0.0208],
    [0.00490, 0.00289, 0.00205, 0.00790, 0.0428],
    [0.0510, 0.0358, 0.0227, 0.00722, 0.0193],
    [0.586, 0.295, 0.217, 0.105, 0.0521],
    [7.2842, 3.893, 2.073, 0.996, 0.3768],
    [98.961, 51.0267, 27.64425, 13.780, 4.172],
]

speedups = []

for size in times:
    speedup = []
    seq = size[0]
    for time in size[1:]:
        speedup.append(seq / time)
    speedups.append(speedup)

speedups = np.array(speedups)
print(speedups)

efficiencies = []

for size in times:
    efficiency = []
    seq = size[0]
    for time, p in zip(size[1:], procs[1:]):
        efficiency.append( seq / (time * p))
    efficiencies.append(efficiency)
efficiencies = np.array(efficiencies)
print(efficiencies)

# Take advantage of downwards broadcasting
kfs = (1 / speedups - 1 / np.array(procs[1:])) / (1 - 1 / np.array(procs[1:]))
print(kfs)
