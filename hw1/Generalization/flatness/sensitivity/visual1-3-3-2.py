import csv 
import matplotlib.pyplot as plt
import numpy as np
row = np.load('record_new.npy')
row = row[row[:,0].argsort()]
print(row)
#plt.xlim(0,2000)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()


plt.xlabel("batch_size")
# ax1.set_ylabel("cross entropy loss")
ax1.set_ylabel("acc")
ax1.plot(row[:,0] , row[:,1] ,	label="training_acc" , color='blue')
ax1.plot(row[:,0] , row[:,4] , '--' , label="testing_acc" , color='blue')
# ax1.plot(row[:,0] , row[:,2] ,	label="training_loss" , color='blue')
# ax1.plot(row[:,0] , row[:,5] , '--' , label="testing_loss" , color='blue')
ax1.legend()
ax2.set_ylabel("sensitivity")
ax2.plot(row[:,0] , row[:,3]   ,label="sensitivity" , color='red')
# ax2.plot(row[:,0] , row[:,6] , '--' ,label="testing_sensitivity" , color='red')
ax2.legend()

fig.tight_layout()
plt.show()
