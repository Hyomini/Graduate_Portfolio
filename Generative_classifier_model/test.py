import numpy as np

mu_hat = [1,2,3,4,5,6,7,8,9]
mu_hat = np.array(mu_hat)
mu_hat = np.reshape(mu_hat, (9,1))

mu_hat1 = [10,7,5,1,6,8,2,3,1]
mu_hat1 = np.array(mu_hat1)
mu_hat1 = np.reshape(mu_hat1, (9,1))

mu_hat2 = mu_hat1-mu_hat
mu_hat2_s = np.matmul(mu_hat2, mu_hat2.T)
mu_hat1_s = np.matmul(mu_hat1, mu_hat1.T)
mu_hat_s = np.matmul(mu_hat, mu_hat.T)
print(mu_hat2_s+mu_hat1_s+mu_hat_s)
#print(np.linalg.inv(mu_hat_s))
print(np.linalg.inv(mu_hat2_s+mu_hat1_s+mu_hat_s))
print(np.linalg.det(mu_hat2_s+mu_hat1_s+mu_hat_s))