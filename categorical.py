import numpy as np
import pandas as pd

# dataset
from sklearn.datasets import load_iris
df_import = load_iris(as_frame=True)
X = df_import['data']
y = df_import['target'].rename('y')
df = pd.concat([X, y], axis=1)

# --- EXPLORATORY DATA ANALYSIS & UNSUPERVISED METHODS

import matplotlib.pyplot as plt
import seaborn as sns

# Class counts
sns.countplot(x=y)
plt.title('Class Frequency')
plt.xlabel('class')
plt.show()

# Feature statistics by class
class_stats = df.groupby('y').agg(['mean', 'std']).T
class_stats.columns.name = 'feature'
class_stats

# Principal Component Analysis
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_PCA = pd.DataFrame(pca.fit_transform(X))
X_PCA.columns = ['PC'+str(n+1) for n in range(X_PCA.shape[1])]

# 2D Scatter Plot
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y)

from sklearn import metrics
def evaluate(y_test, y_pred):

    sorted_labels = sorted(y_test.unique())

    freqs = [sum(y_test==l)/len(y_test) for l in sorted_labels]
    accs = metrics.confusion_matrix(y_test, y_pred, labels=sorted_labels,
        normalize='true').diagonal()
    precs = metrics.precision_score(y_test, y_pred, labels=sorted_labels, average=None)
    recs = metrics.recall_score(y_test, y_pred, labels=sorted_labels, average=None)

    res = pd.DataFrame([freqs, accs, precs, recs],
        index=['Frequency', 'Accuracy', 'Precision', 'Recall']).T
    res.index.name = 'True Class'
    
    avgs = res.mean(axis=0)
    weighted_avgs = [res.iloc[:,i].dot(res.iloc[:,0]) for i in range(res.shape[1])]
    res = (res
        .append(pd.Series(avgs, index=res.columns, name='Class Average'))
        .append(pd.Series(weighted_avgs, index=res.columns, name='Weighted Average'))
        .round(4))
    res.loc[['Class Average', 'Weighted Average'],'Frequency'] = '-'

    print(res, end='\n\n')

    cm = metrics.confusion_matrix(y_test, y_pred, labels=sorted_labels)
    metrics.ConfusionMatrixDisplay(cm, display_labels=sorted_labels).plot()
    plt.title('Confusion Matrix')
    plt.show()

# Supervised Learning Models

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
evaluate(y_test=y_test, y_pred=rf_pred)

# XGBoost
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV 

le = LabelEncoder() 
y_train_le = le.fit_transform(y_train)

param_grid = {
    'gamma': [0.1, 0.2, 0.3],
    'lambda': [0.4, 0.5, 0.6],
    'learning_rate': [0.1, 0.15, 0.2],
    'max_depth': [8, 10, 12]
    }

xgb = GridSearchCV(
    estimator=XGBClassifier(use_label_encoder=False), 
    param_grid=param_grid, cv=5)
xgb.fit(X_train, y_train_le, eval_metric='mlogloss')
cv_results = pd.DataFrame(xgb.cv_results_)
cv_results.loc[:,
    [c for c in cv_results.columns if c[0:6]=='param_'] +
    ['mean_test_score', 'std_test_score']]
xgb_pred = le.inverse_transform(xgb.predict(X_test))
evaluate(y_test=y_test, y_pred=xgb_pred)

# Neural Network
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras import Sequential, layers, optimizers, callbacks

def plot_log_losses(history):
    loss = np.array(history['loss'])
    val_loss = np.array(history['val_loss'])

    epochs = np.array(range(1, len(loss) + 1))

    plt.plot(epochs, np.log(loss+10e-10), label='Training Loss') # smooth for 0
    plt.plot(epochs, np.log(val_loss+10e-10), label='Validation Loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.title(f'Training and Validation Loss (log scale)')
    
    #plt.savefig(f'plots/loss_{model_name}_r{run_idx}.jpg')
    plt.show()

ss = StandardScaler()
X_train_std = ss.fit_transform(X_train)
X_test_std = ss.transform(X_test)
ohe = OneHotEncoder()
y_train_ohe = ohe.fit_transform(y_train.values.reshape(-1, 1)).toarray()
y_test_ohe = ohe.transform(y_test.values.reshape(-1, 1)).toarray()

nn = Sequential()
nn.add(layers.Dense(12, input_shape=(4,)))
nn.add(layers.Dense(12, activation='sigmoid'))
nn.add(layers.Dense(12, activation='relu'))
nn.add(layers.Dense(3, activation='softmax'))
nn.summary()

adam = optimizers.Adam(learning_rate=0.01)
early_stop = callbacks.EarlyStopping(patience=50, restore_best_weights=True)
nn.compile(optimizer=adam, loss='categorical_crossentropy')
hist = nn.fit(X_train_std, y_train_ohe, epochs=400, batch_size=4,
    validation_split=0.2, callbacks=[early_stop], verbose=0)
plot_log_losses(hist.history)
nn_pred = ohe.inverse_transform(nn.predict(X_test_std))
evaluate(y_test=y_test, y_pred=nn_pred)
