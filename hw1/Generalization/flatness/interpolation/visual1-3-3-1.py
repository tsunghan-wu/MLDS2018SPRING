import csv 
import matplotlib.pyplot as plt
import numpy as np
m = 1
row = np.load('./record'+ str(m) + '.npy')
print(row)
#plt.xlim(0,2000)

fig, ax1 = plt.subplots()



plt.xlabel("alpha rate")
ax1.set_ylabel("acc")
ax1.plot(row[:,0] , row[:,1] , label="training_acc" , color='blue')
ax1.plot(row[:,0] , row[:,3] , label="testing_acc" , color='red')
ax1.legend()
# plt.legend()
# fig.tight_layout()
# plt.show()
ax2 = ax1.twinx()
ax2.set_ylabel("cross_entropy")
ax2.plot(row[:,0] , row[:,2] , label="training_cross_entropy" , color='purple')
ax2.plot(row[:,0] , row[:,4] , label="testing_cross_entropy" , color='orange')
ax2.legend(loc=0)

fig.tight_layout()
plt.savefig("./1-3-3-1.png")
