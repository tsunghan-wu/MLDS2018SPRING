import csv 
import matplotlib.pyplot as plt
import numpy as np
cin1 = csv.reader(open('error_table_cnn1_dim50.csv' , 'r'))
cin2 = csv.reader(open('error_table_cnn1_dim48_42.csv' , 'r'))
cin3 = csv.reader(open('error_table_cnn2_dim48_42.csv' , 'r')) 

r1 = np.array([row for row in cin1]).astype(np.float)
r2 = np.array([row for row in cin2]).astype(np.float)
r3 = np.array([row for row in cin3]).astype(np.float)
row1,row2,row3,row4 = [],[],[],[]
span = 30
for _ in range(len(r1) - span):
	s = [0,0,0,0]
	for __ in range(span):
		s[0] += r1[_ + __]
		s[1] += r2[_ + __]
		s[2] += r3[_ + __]
	row1 .append( s[0] / span )
	row2 .append( s[1] / span )
	row3 .append( s[2] / span )
row1 = np.array(row1)
row2 = np.array(row2)
row3 = np.array(row3)

print(row1)
#plt.xlim(0,2000)
plt.ylabel("log2(training_loss)")
plt.xlabel("epoch(30mean)")
plt.plot(row1[:,0] , np.log(row1[:,2]) , label="error_table_cnn1_dim50")
plt.plot(row2[:,0] , np.log(row2[:,2]) , label="error_table_cnn1_dim48_42")
plt.plot(row3[:,0] , np.log(row3[:,2]) , label="error_table_cnn2_dim48_42")
plt.legend()
plt.show()


