import csv 
import matplotlib.pyplot as plt
import numpy as np
cin1 = csv.reader(open('model_1layers/error_table_1layers_test.csv' , 'r'))
cin2 = csv.reader(open('model_3layers/error_table_3layers.csv' , 'r'))
cin3 = csv.reader(open('model_5layers/error_table_5layers.csv' , 'r')) 

row1 = np.array([row for row in cin1]).astype(np.float)
row2 = np.array([row for row in cin2]).astype(np.float)
row3 = np.array([row for row in cin3]).astype(np.float)



plt.plot(row1[:,0] , np.log(row1[:,1]) , color='blue' , label="DNN_with_1layer")
plt.plot(row3[:,0] , np.log(row3[:,1]) , color='red' , label="DNN_with_5layer")
plt.plot(row2[:,0] , np.log(row2[:,1]) , color='green' , label="DNN_with_3layer")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("log2(loss)")
plt.show()


