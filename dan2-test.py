import dan2
import numpy as np
import pickle


# Test dataset for classification
with open('data/x-train.P', 'rb') as f:
    X_train = pickle.load(f)
    X_train = np.array(X_train)

with open('data/y-train.P', 'rb') as f:
    y_train = pickle.load(f)
    y_train = np.array(y_train)
    #y_train = np.where(y_train==1, 100, -100)

print(X_train)
y_train = y_train.reshape(len(y_train), 1)
print(y_train)

clf = dan2.DAN2Regressor(depth=10)
clf.fit(X_train, y_train, f_0 = None)

print('model_weights', clf.model['weights'])
print('model_intercepts', clf.model['intercept'])
print('model_mu', clf.model['mu'])