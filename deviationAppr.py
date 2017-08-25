import numpy as np
import matplotlib
import matplotlib.pyplot as plt

x = np.arange(40)
y = np.log(x + 1) * np.exp(-x/8.) * x**2 + np.random.random(40) * 15
rft = np.fft.rfft(y)
rft[10:] = 0
y_smooth = np.fft.irfft(rft)

plt.plot(x, y, label='Original')
plt.plot(x, y_smooth, label='Smoothed')
plt.legend(loc=0).draggable()
plt.show()