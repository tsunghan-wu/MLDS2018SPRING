import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt('1-2-3recorder')
print(data)
#plt.ylim(0,0.01)
plt.xlabel('minimal ratio')
plt.ylabel('loss')
plt.scatter(data[:,0] , data[:,1])
plt.show()	