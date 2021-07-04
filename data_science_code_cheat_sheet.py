import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
import seaborn as sns

# -1. CUSTOM FUNCTIONS

def all_but(all_columns, left_out):
    return [c for c in all_columns if c not in left_out]

def ft_decomp(df): # separate continuous, binary and categorical features

    dtypes = df.dtypes
    decomp = {'continuous': [], 'categorical': [], 'binary': [], 'others': []}
    for c in df.columns:
        if dtypes[c] in ('float64', 'int64'):
            uniques_set = set(df[c].unique())
            if uniques_set == set([0, 1]):
                decomp['binary'].append(c)
            else:
                decomp['continuous'].append(c)
        elif dtypes[c] == 'object':
            decomp['categorical'].append(c)
        else:
            decomp['others'].append(c)

    return decomp


# 0. DATASET
from sklearn.datasets import load_boston
boston_import = load_boston()
X = pd.DataFrame(boston_import['data'], columns=boston_import['feature_names'])
X['CAT'] = np.random.choice( # categorical noise
    ['A', 'B', 'C'], 
    size=X.shape[0],
    p=[1/3, 1/3, 1/3])
y = pd.Series(boston_import['target'], name='MEDV')
data = pd.concat([X, y], axis=1)


# 1. REGRESSION PROBLEM

# --- 1.1 EXPLORATORY DATA ANALYSIS & UNSUPERVISED METHODS

data.describe()

# Target variable distribution
sns.histplot(x=data['MEDV'])
plt.title('Histogram of House Median Value')
plt.xlabel('MEDV')
plt.show()

# Scatterplots between each feature and the target
for c in X.columns:
    sns.scatterplot(x=X[c], y=y)
    plt.show()


# --- 1.2 SUPERVISED METHODS

# ------ 1.2.1 Split and Preprocessing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

decomp = ft_decomp(X_train)
transformers = [
    ('standard_scaler', StandardScaler(), decomp['continuous']),
    ('ohe', OneHotEncoder(), decomp['categorical']),
    ('passthrough', 'passthrough', decomp['binary']+decomp['others'])]
ct = ColumnTransformer(transformers)
X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)

# ------ 1.2.2 Evaluation Metrics
from sklearn import metrics
def reg_evaluate(y_test, y_pred):

    # true vs predicted plot
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, s=10)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()])]  # max of both axes

    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title('Test set results')
    plt.show()

    # eval metrics
    rec_metrics = dict()
    rec_metrics['R-Squared'] = metrics.r2_score(y_test, y_pred)
    rec_metrics['MAE'] = metrics.mean_absolute_error(y_test, y_pred)
    rec_metrics['MSE'] = metrics.mean_squared_error(y_test, y_pred)

    for m in rec_metrics:
        print(f'{m} = {round(rec_metrics[m], 4)}')

# ------ 1.2.3 Supervised Learning Models

print('\nLinear Regression')
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
reg_evaluate(y_test, lr_pred)

print('\nLasso')
from sklearn.linear_model import LassoCV
lasso = LassoCV(cv=5)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
reg_evaluate(y_test, lasso_pred)

print('\nRandom Forest')
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=200, criterion='mse')
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
reg_evaluate(y_test=y_test, y_pred=rf_pred)

print('\nXGBoost')
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV 

param_grid = {
    'gamma': [0.1, 0.2, 0.3],
    'lambda': [0.4, 0.5, 0.6],
    'learning_rate': [0.1, 0.15, 0.2],
    'max_depth': [8, 10, 12]
    }

xgb = GridSearchCV(
    estimator=XGBRegressor(objective='reg:squarederror'), 
    param_grid=param_grid, cv=5)
xgb.fit(X_train, y_train)
cv_results = pd.DataFrame(xgb.cv_results_)
cv_results.loc[:,
    [c for c in cv_results.columns if c[0:6]=='param_'] +
    ['mean_test_score', 'std_test_score']]
xgb_pred = xgb.predict(X_test)
reg_evaluate(y_test=y_test, y_pred=xgb_pred)

print('\nNeural Network')
from tensorflow.keras import Sequential, layers, optimizers, callbacks

def plot_losses(history):
    
    loss = np.array(history['loss'])
    val_loss = np.array(history['val_loss'])
    log_loss = np.log(loss)
    log_val_loss = np.log(val_loss)
    epochs = np.array(range(1, len(loss) + 1))

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(epochs, loss, label='training loss')
    axs[0].plot(epochs, val_loss, label='validation loss')
    axs[0].legend(loc='upper right')
    axs[0].set_xlabel('epoch')

    axs[1].plot(epochs, log_loss, label='training log loss')
    axs[1].plot(epochs, log_val_loss, label='validation log loss')
    axs[1].legend(loc='upper right')
    axs[1].set_xlabel('epoch')
    plt.show()

reg_nn = Sequential(name='reg_nn')
reg_nn.add(layers.Dense(16, input_shape=(16,)), activation='sigmoid')
reg_nn.add(layers.Dense(16, activation='sigmoid'))
reg_nn.add(layers.Dense(16, activation='sigmoid'))
reg_nn.add(layers.Dense(1, activation='linear'))
reg_nn.summary()

adam = optimizers.Adam(learning_rate=0.01)
early_stop = callbacks.EarlyStopping(patience=50, restore_best_weights=True)
reg_nn.compile(optimizer=adam, loss='mean_squared_error')
s = time()
hist = reg_nn.fit(verbose=0,
    x=X_train, y=y_train, epochs=400, batch_size=8, callbacks=[early_stop],
    validation_split=0.2, shuffle=False) # already shuffled, reproducible
t = time() - s
print(f'Training time: {round(t, 0)}s ({round(t/60, 2)}m)')
plot_losses(hist.history)
reg_nn_pred = reg_nn.predict(X_test)
reg_evaluate(y_test=y_test, y_pred=reg_nn_pred)


# 2. CLASSIFICATION PROBLEM

# adapting the problem to classification
def class_func(x):
    if x < 20:
        o = 'low'
    elif x < 40:
        o = 'medium'
    else:
        o = 'high'

    return o
category = data['MEDV'].apply(class_func).rename('category')
data['category'] = category
X = data.iloc[:,:-2]
y = data.loc[:,'category'].rename('y') # leaving 'MEDV' out


# --- 2.1 EXPLORATORY DATA ANALYSIS & UNSUPERVISED METHODS

import matplotlib.pyplot as plt
import seaborn as sns

# Class counts
sns.countplot(x=y)
plt.title('Class frequency')
plt.ylabel('Count')
plt.xlabel('Class')
plt.show()

# Feature statistics by class
class_stats = data.groupby('category').agg(['mean', 'std']).T
class_stats.columns.name = 'feature'
class_stats

# Principal Component Analysis (standardized)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

pca_std = Pipeline([
    ('scaling', StandardScaler()),
    ('pca', PCA(n_components=2))])
cont_cols = ft_decomp(X)['continuous']
X_PCA = pd.DataFrame(pca_std.fit_transform(X[cont_cols]))
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
k_means_clusters = k_means.fit_predict(X[cont_cols])

# todo clustering evaluation


# --- 2.2 SUPERVISED METHODS

# ------ 2.2.1 Split and Preprocessing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# trees are scale invariant, but neural networks require standardization
# in theory, trees do use (non-binary) categorical features
# however, sklearn's and xgboost's implementations in python do not support this

decomp = ft_decomp(X_train)
transformers = [
    ('standard_scaler', StandardScaler(), decomp['continuous']),
    ('ohe', OneHotEncoder(), decomp['categorical']),
    ('passthrough', 'passthrough', decomp['binary']+decomp['others'])]
ct = ColumnTransformer(transformers)
X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)

# ------ 2.2.2 Evaluation Metrics
from sklearn import metrics
def clf_evaluate(y_test, y_pred):

    sorted_labels = sorted(y_test.unique())

    # confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred, labels=sorted_labels)
    metrics.ConfusionMatrixDisplay(cm, display_labels=sorted_labels).plot()
    plt.title('Confusion Matrix')
    plt.show()

    freqs = [sum(y_test==l)/len(y_test) for l in sorted_labels]
    accs = metrics.confusion_matrix(y_test, y_pred, labels=sorted_labels,
        normalize='true').diagonal()
    precs = metrics.precision_score(y_test, y_pred, labels=sorted_labels, average=None)
    recs = metrics.recall_score(y_test, y_pred, labels=sorted_labels, average=None)

    res = pd.DataFrame([freqs, accs, precs, recs],
        index=['Frequency', 'Accuracy', 'Precision', 'Recall'],
        columns=sorted_labels).T
    res.index.name = 'True Class'
    res = res.sort_values(by='Frequency', ascending=False)
    
    avgs = res.mean(axis=0)
    weighted_avgs = [res.iloc[:,i].dot(res.iloc[:,0]) for i in range(res.shape[1])]
    res = (res
        .append(pd.Series(avgs, index=res.columns, name='Class Average'))
        .append(pd.Series(weighted_avgs, index=res.columns, name='Weighted Average'))
        .round(4))
    res.loc[['Class Average', 'Weighted Average'],'Frequency'] = '-'
    
    print(f'\nAccuracy = {round(res.loc["Weighted Average", "Accuracy"], 4)}')
    print('\n', res)

# ------ 2.2.3 Supervised Learning Models

print('\nRandom Forest')
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, criterion='gini')
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
clf_evaluate(y_test=y_test, y_pred=rf_pred)

print('\nXGBoost')
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
xgb.fit(X_train, y_train_le, eval_metric='mlogloss') # 'merror' for binary clf
cv_results = pd.DataFrame(xgb.cv_results_)
cv_results.loc[:,
    [c for c in cv_results.columns if c[0:6]=='param_'] +
    ['mean_test_score', 'std_test_score']]
xgb_pred = le.inverse_transform(xgb.predict(X_test))
clf_evaluate(y_test=y_test, y_pred=xgb_pred)

print('\nNeural Network')
from tensorflow.keras import Sequential, layers, optimizers, callbacks

def plot_losses(history):
    
    loss = np.array(history['loss'])
    val_loss = np.array(history['val_loss'])
    log_loss = np.log(loss)
    log_val_loss = np.log(val_loss)
    epochs = np.array(range(1, len(loss) + 1))

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(epochs, loss, label='training loss')
    axs[0].plot(epochs, val_loss, label='validation loss')
    axs[0].legend(loc='upper right')
    axs[0].set_xlabel('epoch')

    axs[1].plot(epochs, log_loss, label='training log loss')
    axs[1].plot(epochs, log_val_loss, label='validation log loss')
    axs[1].legend(loc='upper right')
    axs[1].set_xlabel('epoch')
    plt.show()

# X is already scaled and one-hot-encoded (categ. features)
# however, y must be one-hot-encoded for the softmax layer
ohe = OneHotEncoder()
y_train_ohe = ohe.fit_transform(y_train.values.reshape(-1, 1)).toarray()

clf_nn = Sequential(name='clf_nn')
clf_nn.add(layers.Dense(16, activation='relu', input_shape=(16,)))
clf_nn.add(layers.Dense(16, activation='relu'))
clf_nn.add(layers.Dense(3, activation='softmax'))
clf_nn.summary()

adam = optimizers.Adam(learning_rate=0.01)
early_stop = callbacks.EarlyStopping(patience=50, restore_best_weights=True)
clf_nn.compile(optimizer=adam, loss='categorical_crossentropy')
hist = clf_nn.fit(verbose=0,
    x=X_train, y=y_train_ohe, epochs=400, batch_size=8, callbacks=[early_stop],
    validation_split=0.2, shuffle=False) # already shuffled, reproducible
plot_losses(hist.history)
clf_nn_pred = ohe.inverse_transform(clf_nn.predict(X_test))
clf_evaluate(y_test=y_test, y_pred=clf_nn_pred)
