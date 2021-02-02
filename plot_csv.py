#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys

for csv_file in sys.argv[1:]:
    data = np.genfromtxt(csv_file, delimiter=',', names=['x', 'y'])
    print("data:", data)
    plt.plot(data['x'], data['y'])
    plt.show()
