import csv 
import matplotlib.pyplot as plt
import numpy as np
rows = [ row.split(',') for row in open('recorder','r')]
data = []
for row in rows:
	this_row  = []
	for cell in row:
		this_row.append(cell.split()[-1])
	data.append(this_row)
row = np.array(data).astype(float)
#plt.xlim(0,2000)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

plt.xlabel("batch_size")
ax1.set_ylabel("loss")
ax1.plot(row[:,0] , row[:,-4] ,	label="training_loss" , color='blue')
ax1.plot(row[:,0] , row[:,-2] , '--' , label="testing_loss" , color='blue')
ax2.set_ylabel("sharpness")
ax2.plot(row[:,0] , row[:,1]   ,label="sharpness(random_loss)" , color='red')
#ax2.plot(row[:,0] , row[:,5] , '--' ,label="testing_sensitivity" , color='red')
ax1.legend()
ax2.legend()
fig.tight_layout()
plt.show()
