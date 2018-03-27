import matplotlib.pyplot as plt
import numpy as np
file = ['1-2_loss.csv','1-2_grad_norm.csv'][1]
raw = np.array([r for r in open(file,'r')]).astype(np.float)
data = []
mean = 50
for i in range(len(raw)-mean):
	data.append(np.mean(raw[i:i+mean]))
print(data)
plt.xlabel('epoch(50 mean)')
plt.ylabel(file)
plt.plot(data)
plt.show()