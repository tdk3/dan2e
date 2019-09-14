import dan2
import numpy as np
import pickle
import sys
'''
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
'''



def load_file(path):
    with open(path, 'rb') as f:
        
        file = pickle.load(f)

        if type(file) is not np.ndarray:
            file = np.array(file)

        return file


def save_file(path, model):
    with open(path + '.pk', 'wb') as save_file:
        pickle.dump(model, save_file)


def main(X, y, depth):
    clf = dan2.DAN2Regressor(depth=depth)
    clf.fit(X, y)
    path = clf.name
    save_file(path, clf)
    print(clf.coef_)
    y_pred = clf.predict(X)
    




if __name__ == '__main__':
    X = load_file(sys.argv[1])
    y = load_file(sys.argv[2])
    y = y.reshape(len(y), 1)
    main(X, y, int(sys.argv[3]))

'''
print('model_weights', clf.model['weights'])
print('model_intercepts', clf.model['intercept'])
print('model_mu', clf.model['mu'])
'''