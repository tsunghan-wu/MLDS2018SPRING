import csv 
import matplotlib.pyplot as plt
import numpy as np
row = np.load('1-3-3-2_record.npy')
row = row[row[:,0].argsort()]
print(row)
#plt.xlim(0,2000)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()


plt.xlabel("batch_size")
ax1.set_ylabel("acc")
ax1.plot(row[:,0] , row[:,1] ,	label="training_acc" , color='blue')
ax1.plot(row[:,0] , row[:,4] , '--' , label="testing_acc" , color='blue')
ax2.set_ylabel("sensitivity")
ax2.plot(row[:,0] , row[:,3]   ,label="training_sensitivity" , color='red')
ax2.plot(row[:,0] , row[:,5] , '--' ,label="testing_sensitivity" , color='red')
plt.legend()

fig.tight_layout()
plt.show()
