import numpy as np

a = np.array(range(10000))
f = open('temp.txt','w')

for i in range(10000):
    f.write(str(a[i]))