# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


a = np.random.randint(-10, 10, size=(5, 5, 3))
a = np.asarray(a, dtype='float64')

print(a)

b = np.reshape(a, (5, -1))

print(b)