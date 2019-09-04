import matplotlib.pyplot as plt
import numpy as np

def make_plot(ite,mahala_axis,euc_axis) :
  plt.title("Time Comparison(mnist test set) - on PU")
  plt.xlabel('Iteration')
  plt.ylabel('Time')
  plt.plot(ite, mahala_axis, c="r", label="mahalanobis")
  plt.plot(ite, euc_axis, c="b", label="euclidean")
  plt.legend(loc=5)
  plt.show()