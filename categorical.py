import pandas as pd
from pandas.tseries import frequencies

# dataset
from sklearn.datasets import load_iris
df_import = sklearn.datasets.load_iris(as_frame=True)
X = df_import['data']
y = df_import['target'].rename('y')
df = pd.concat([X, y], axis=1)

# --- VISUALIZATION & UNSUPERVISED METHODS

# Principal Component Analysis
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_PCA = pd.DataFrame(pca.fit_transform(X))
X_PCA.columns = ['PC'+str(n+1) for n in range(X_PCA.shape[1])]

# 2D Scatter Plot
import matplotlib.pyplot as plt
import seaborn as sns
# palettes: https://seaborn.pydata.org/tutorial/color_palettes.html
sns.scatterplot(
    data=pd.concat((X_PCA, y), axis=1),
    x='PC1', y='PC2', hue='y', palette='colorblind')
plt.title('First two principle components colored by class')
plt.legend(title='class')
plt.show()

# Clustering
from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=3)
k_means_clusters = k_means.fit_predict(X)

# todo develop this

# --- SUPERVISED METHODS

# Split and Evaluation Function
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn import metrics
def evaluate(y_test, y_pred):

    sorted_labels = sorted(y_test.unique())

    freqs = [sum(y_test==l)/len(y_test) for l in sorted_labels]
    accs = metrics.confusion_matrix(y_test, y_pred, labels=sorted_labels,
        normalize='true').diagonal()
    precs = metrics.precision_score(y_test, y_pred, labels=sorted_labels, average=None)
    recs = metrics.recall_score(y_test, y_pred, labels=sorted_labels, average=None)

    res = pd.DataFrame([freqs, accs, precs, recs],
        index=['Frequency', 'Accuracy', 'Precision', 'Recall']).T.round(4)
    res.index.name = 'True Class'
    
    avgs = res.mean(axis=0)
    weighted_avgs = [res.iloc[:,i].dot(res.iloc[:,0]) for i in range(res.shape[1])]
    res = res.append(pd.Series(avgs, index=res.columns, name='Class Average'))
    res = res.append(pd.Series(weighted_avgs, index=res.columns, name='Weighted Average'))
    res.loc[['Class Average', 'Weighted Average'],'Frequency'] = '-'
    return res

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
evaluate(y_test=y_test, y_pred=rf_pred)