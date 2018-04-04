import csv 
import matplotlib.pyplot as plt
import numpy as np
cin = csv.reader(open('fake_loss.csv' , 'r'))

r = np.array([row for row in cin]).astype(np.float)
row = []
span = 1
for _ in range(len(r) - span):
	s = 0
	for __ in range(span):
		s += r[_ + __]
	row .append( s / span )
row = np.array(row)


print(row)
#plt.xlim(0,2000)
plt.ylabel("training_acc")
plt.xlabel("epoch")
plt.scatter(row[:,0] , row[:,1] , label="training_acc" , color='blue',s=2)
plt.scatter(row[:,0] , row[:,3] , label="testing_acc" , color='red',s=2)
plt.legend()
plt.show()

plt.cla()
plt.ylabel("training_loss")
plt.xlabel("epoch")
plt.scatter(row[:,0] , row[:,2] , label="training_loss" , color='blue',s=2)
plt.scatter(row[:,0] , row[:,4] , label="testing_loss" , color='red',s=2)
plt.legend()
plt.show()

