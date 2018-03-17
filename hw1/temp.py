import csv 
import matplotlib.pyplot as plt
import numpy as np
cin1 = csv.reader(open('error_table_cnn2_dim256.csv' , 'r'))
cin2 = csv.reader(open('error_table_cnn2_dim128_128_128.csv' , 'r'))
cin3 = csv.reader(open('error_table_cnn4_dim128_128_128.csv' , 'r')) 
cin4 = csv.reader(open('error_table_residual_cnn4_dim128_128_128.csv' , 'r')) 

r1 = np.array([row for row in cin1]).astype(np.float)
r2 = np.array([row for row in cin2]).astype(np.float)
r3 = np.array([row for row in cin3]).astype(np.float)
r4 = np.array([row for row in cin4]).astype(np.float)
row1,row2,row3,row4 = [],[],[],[]
span = 30
for _ in range(len(r1) - span):
	s = [0,0,0,0]
	for __ in range(span):
		s[0] += r1[_ + __]
		s[1] += r2[_ + __]
		s[2] += r3[_ + __]
		s[3] += r4[_ + __]
	row1 .append( s[0] / span )
	row2 .append( s[1] / span )
	row3 .append( s[2] / span )
	row4 .append( s[3] / span )
row1 = np.array(row1)
row2 = np.array(row2)
row3 = np.array(row3)
row4 = np.array(row4)


print(row1)
plt.ylabel("log2(loss)")
plt.xlabel("epoch(30mean)")
plt.plot(row1[:,0] , np.log(row1[:,2]) , label="error_table_cnn2_dim256")
plt.plot(row2[:,0] , np.log(row2[:,2]) , label="error_table_cnn2_dim128_128_128")
plt.plot(row3[:,0] , np.log(row3[:,2]) , label="error_table_cnn4_dim128_128_128")
plt.plot(row4[:,0] , np.log(row4[:,2]) , label="error_table_residual_cnn4_dim128_128_128")
plt.legend()
plt.show()


