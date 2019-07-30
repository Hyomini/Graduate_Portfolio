import matplotlib.pyplot as plt
import numpy as np

def make_plot(ite,mahalanobis_dist,euclidean_dist) :

  min_temp = min(np.min(mahalanobis_dist),np.min(euclidean_dist))
  max_temp = max(np.max(mahalanobis_dist),np.max(euclidean_dist))
  x_value = np.arange(1,ite+1,1)

  plt.title("Distance ComPare")
  plt.xlabel('Iteration')
  plt.ylabel('Distance')
  plt.scatter(x_value,mahalanobis_dist)
  plt.scatter(x_value,euclidean_dist)
  plt.plot(x_value,mahalanobis_dist,x_value,euclidean_dist, '-r')
  plt.legend(['Mahalanobis', 'Euclidean'])
  plt.ylim(min_temp,max_temp)
  plt.xticks(x_value)
  plt.grid()
  plt.show()