import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from sklearn.cross_validation import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as ss
from sklearn.decomposition import PCA

x = np.arange(0, 10)
xU, xL = x + 0.5, x - 0.5
prob = ss.norm.cdf(xU, scale = 3) - ss.norm.cdf(xL, scale = 3)
prob = prob / prob.sum()
X = np.random.choice(x, size = 10000, p = prob)
print(X)

Y1 = np.random.normal(5, 1.5, 3000)
Y1 = X[:3000] + Y1
Y1 = pd.DataFrame(Y1)

Y2 = np.random.uniform(low=0, high=7)
Y2 = X[3000:5000] + Y2
Y2 = pd.DataFrame(Y2)

Y3 = X[5000:] + 0.5
Y3 = pd.DataFrame(Y3)


Y = pd.concat([Y1, Y2, Y3], axis = 0, names = ['Y'])

# plot.scatter(X, Y)
# plot.show()

X_without_drop = pd.get_dummies(X, prefix='d', drop_first=False)
X_drop_first = pd.get_dummies(X, prefix='d', drop_first= True)

df_random = pd.DataFrame(np.random.randn(10000, 30), columns=list(range(1,31, 1)))

X_without_drop = pd.concat([X_without_drop, df_random], axis = 1)
X_drop_first = pd.concat([X_drop_first, df_random], axis = 1)

# X WITHOUT DROP FIRST-----------------------------------------------------------------------------------------------

# SIN PCA
xTrain, xTest, yTrain, yTest = train_test_split(X_without_drop,Y,test_size=0.3,random_state=42)
mseOos = []

nTreeList = range(1, 20, 1)
for iTrees in nTreeList:
    depth = None
    maxFeat = 13
    fileModel = ensemble.RandomForestRegressor(criterion='mse', bootstrap=False,min_samples_leaf=20,
                                               min_samples_split=200,n_estimators=iTrees, max_depth=depth,
                                               max_features= maxFeat, oob_score=False, random_state= 531)
    fileModel.fit(xTrain,yTrain)
    prediction = fileModel.predict(xTest)
    mseOos.append(mean_squared_error(yTest, prediction))


[without_pca] = plot.plot(nTreeList, mseOos)
plot.xlabel('Number of Trees in Ensemble')
plot.ylabel('MSE')
plot.ylim([0.0, 1.1*max(mseOos)])
# plot.show()

# CON PCA
pca = PCA(whiten=True, svd_solver='randomized')
pca.fit(X_without_drop)
var = pca.explained_variance_ratio_
pca_components = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4) * 100)
# plot.plot(pca_components)
#plot.show()

pca_components = 29
pca = PCA(n_components=pca_components, whiten=True, svd_solver='randomized')
pca.fit(X_without_drop)
X_without_drop = pca.fit_transform(X_without_drop)
X_without_drop = pd.DataFrame(X_without_drop)

xTrain, xTest, yTrain, yTest = train_test_split(X_without_drop,Y,test_size=0.3,random_state=42)
mseOos = []


for iTrees in nTreeList:
    depth = None
    maxFeat = 9
    fileModel = ensemble.RandomForestRegressor(criterion='mse', bootstrap=False,min_samples_leaf=20,
                                               min_samples_split=200,n_estimators=iTrees, max_depth=depth,
                                               max_features= maxFeat, oob_score=False, random_state= 531)
    fileModel.fit(xTrain, yTrain)
    prediction = fileModel.predict(xTest)
    mseOos.append(mean_squared_error(yTest, prediction))


[pca_imp] = plot.plot(nTreeList, mseOos)
plot.xlabel('Number of Trees in Ensemble')
plot.ylabel('MSE')
plot.ylim([0.0, 1.1*max(mseOos)])



# X WITHOUT DROP FIRST-----------------------------------------------------------------------------------------------

# SIN PCA
xTrain, xTest, yTrain, yTest = train_test_split(X_drop_first,Y,test_size=0.3,random_state=42)
mseOos = []

nTreeList = range(1, 20, 1)
for iTrees in nTreeList:
    depth = None
    maxFeat = 13
    fileModel = ensemble.RandomForestRegressor(criterion='mse', bootstrap=False,min_samples_leaf=20,
                                               min_samples_split=200,n_estimators=iTrees, max_depth=depth,
                                               max_features= maxFeat, oob_score=False, random_state= 531)
    fileModel.fit(xTrain,yTrain)
    prediction = fileModel.predict(xTest)
    mseOos.append(mean_squared_error(yTest, prediction))


[without_pca_drop] = plot.plot(nTreeList, mseOos)
plot.xlabel('Number of Trees in Ensemble')
plot.ylabel('MSE')
plot.ylim([0.0, 1.1*max(mseOos)])
# plot.show()

# CON PCA
pca = PCA(whiten=True, svd_solver='randomized')
pca.fit(X_drop_first)
var = pca.explained_variance_ratio_
pca_components = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4) * 100)
print(pca_components)
# plot.plot(pca_components)
# plot.show()

pca_components = 29
pca = PCA(n_components=pca_components, whiten=True, svd_solver='randomized')
pca.fit(X_drop_first)
X_drop_first = pca.fit_transform(X_drop_first)
X_drop_first = pd.DataFrame(X_drop_first)

xTrain, xTest, yTrain, yTest = train_test_split(X_drop_first,Y,test_size=0.3,random_state=42)
mseOos = []


for iTrees in nTreeList:
    depth = None
    maxFeat = 9
    fileModel = ensemble.RandomForestRegressor(criterion='mse', bootstrap=False,min_samples_leaf=20,
                                               min_samples_split=200,n_estimators=iTrees, max_depth=depth,
                                               max_features= maxFeat, oob_score=False, random_state= 531)
    fileModel.fit(xTrain, yTrain)
    prediction = fileModel.predict(xTest)
    mseOos.append(mean_squared_error(yTest, prediction))


[pca_imp_drop] = plot.plot(nTreeList, mseOos)
plot.legend([without_pca, pca_imp, without_pca_drop, pca_imp_drop],
            ['without PCA', 'PCA', 'without PCA drop_first', 'PCA drop_first'], loc = 1)
plot.xlabel('Number of Trees in Ensemble')
plot.ylabel('MSE')
plot.ylim([0.0, 1.1*max(mseOos)])
plot.show()


