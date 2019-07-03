# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

x = np.arange(0, 10, 0.01)

t = stats.t.pdf(x, 1)
n = stats.norm.pdf(x, 0,1)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "26"

plt.plot(x, t, color='orange', label='t-distribution')
plt.plot(x, n,color='steelblue', label='Gaussian distribution ')

plt.hlines(0.2, 0, 10, 'r', 'dashed')
plt.hlines(0.025, 0, 10, 'r', 'dashed')

plt.vlines(0.77, 0, 0.2, 'orange', 'dashed')
plt.vlines(1.17, 0, 0.2, 'steelblue', 'dashed')

plt.vlines(3.42, 0, 0.025, 'orange', 'dashed')
plt.vlines(2.35, 0, 0.025, 'steelblue', 'dashed')

plt.legend(loc='upper right')
plt.xlabel('Distance')
plt.ylabel('Probability ')
plt.ylim(0,0.4)
plt.xlim(0,6)

#ax = plt.gca()
#ax.spines['right'].set_color('none')
#ax.spines['top'].set_color('none')
#ax.xaxis.set_ticks_position('bottom')
#ax.spines['bottom'].set_position(('data', 0))
#ax.yaxis.set_ticks_position('left')
#ax.spines['left'].set_position(('data', 0))

plt.show()
