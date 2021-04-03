import matplotlib.pyplot as plt

from test1 import float_data

temp = float_data[:, 1]
#plt.plot(range(len(temp)), temp)
plt.plot(range(1440), temp[:1440  ])
plt.show()
