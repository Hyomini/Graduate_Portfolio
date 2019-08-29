import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# iris 데이터
iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

#print(iris)
# 결과물 재현을 위해 seed를 설정
np.random.seed(1)

# iris 데이터를 matrix로 변환시킨 후 t-SNE 적용
iris_matrix = iris.iloc[:, 0:4].values
print(iris_matrix)
iris_tsne_result = TSNE(learning_rate=300, init='pca').fit_transform(iris_matrix)


df = pd.DataFrame(iris_tsne_result, columns=('x','y'))
data_points = df.values
kmeans = KMeans(n_clusters=2).fit(data_points)
df['cluster_id']=kmeans.labels_

sns.lmplot('x','y', data=df, fit_reg=False, scatter_kws={"s":500}, hue="cluster_id")
plt.title('kmean plot')
plt.xlabel('x')
plt.ylabel('y')
plt.show()