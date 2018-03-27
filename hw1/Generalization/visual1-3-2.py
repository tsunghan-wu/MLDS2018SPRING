import csv 
import matplotlib.pyplot as plt
import numpy as np
cin = csv.reader(open('1-3-2_few_v1.csv' , 'r'))
row = np.array([row for row in cin]).astype(np.float)

cin = csv.reader(open('1-3-2_few_v2.csv' , 'r'))
temp = np.array([row for row in cin]).astype(np.float)
row = np.append( row,temp ,axis = 0)

cin = csv.reader(open('1-3-2_few_v3.csv' , 'r'))
temp = np.array([row for row in cin]).astype(np.float)
row = np.append( row,temp ,axis = 0)

cin = csv.reader(open('1-3-2_few_v4.csv' , 'r'))
temp = np.array([row for row in cin]).astype(np.float)
row = np.append( row,temp ,axis = 0)

cin = csv.reader(open('1-3-2_v1.csv' , 'r'))
temp = np.array([row for row in cin]).astype(np.float)
row = np.append( row,temp ,axis = 0)

cin = csv.reader(open('1-3-2_v2.csv' , 'r'))
temp = np.array([row for row in cin]).astype(np.float)
row = np.append( row,temp ,axis = 0)

cin = csv.reader(open('1-3-2_v3.csv' , 'r'))
temp = np.array([row for row in cin]).astype(np.float)
row = np.append( row,temp ,axis = 0)

temp = []
for r in row:
	if r[0] < 50000:
		temp.append(r)
row = np.array(temp)
#plt.xlim(0,2000)

s = 10
plt.ylabel("training_acc")
plt.xlabel("number of params")
plt.scatter(row[:,0] , row[:,1] , label="training_acc" , color='blue',s=s)
plt.scatter(row[:,0] , row[:,3] , label="testing_acc" , color='red',s=s)
plt.legend()
plt.show()

plt.cla()
plt.ylabel("training_loss")
plt.xlabel("number of params")
plt.scatter(row[:,0] , row[:,2] , label="training_loss" , color='blue',s=s)
plt.scatter(row[:,0] , row[:,4] , label="testing_loss" , color='red',s=s)
plt.legend()
plt.show()