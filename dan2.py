import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class DAN2Regressor(object):

    def __init__(self, depth=10):
        #super(ClassName, self).__init__()
        self.depth = depth
        self.layers = None
        self.lr = None
        self.f_k = None
        self.A = None
        self.alpha = None
        self.a = None
        self.model = {
            'weights': None,
            'intercept': None,
            'mu': None
        }

    """ Layer activation """
    def f(self, x):
        
        f = self.f_k
        A = self.A
        alpha = self.alpha
        a = self.a
        rows = f.shape[0]
        ''' check if intercept term should be placed first'''
        #Xn = np.hstack((a, A[0]*f, A[1]*np.cos(alpha*x), A[2]*np.sin(alpha*x)))
        Xn = a + A[0]*f + A[1]*np.cos(alpha*x) + A[2]*np.sin(alpha*x)
        return np.sum(Xn)

    """ Method to get alpha column for DAN2 """
    def compute_alpha(self, X):
        cols = X.shape[1]

        """ Create resultant vector of ones """
        R = np.ones(cols)
        print('R', R.shape)

        """ Compute dot product """
        X_dot_R = (1 + np.dot(X,R))
        print('XdR', X_dot_R.shape)
        X_dot_R = X_dot_R.reshape((len(X),))
        print('XdR', X_dot_R.shape)

        """ Compute X and R magnitudes """
        X_mag = np.sqrt(1*1 + np.sum(np.square(X), axis=1))
        R_mag = np.sqrt(np.sum(R**2) + 1*1)

        """ Compute arccosine """
        acos = np.arccos(X_dot_R / (X_mag * R_mag))
        print('acos', acos.shape)

        return acos.reshape(len(acos),1) 


    """ Linear method """
    def linear_reg(self, X, y):
        self.lr = LinearRegression(fit_intercept=True).fit(X, y)
        '''
        if logistic is not True:
            pred = self.lr.predict(X)
        else:
            pred = self.lr.predict(X)
            pred = np.where(pred >= 0.5, 1, 0)'''
        return self.lr.predict(X), self.lr.coef_[0], self.lr.intercept_


    ''' '''
    def build_X1(self, f, alpha, mu=1):
        print('f', f.shape)
        print('alpha', alpha.shape)

        return np.hstack((f, np.cos(alpha*mu), np.sin(alpha*mu)))


    ''' '''
    def build_Xn(self, f, A, a, alpha, mu):
        rows = f.shape[0]
        return np.hstack((A[0]*f, A[1]*np.cos(alpha*mu), A[2]*np.sin(alpha*mu)))


    def logging(self, weights, intercept, mu):

        if self.model['weights'] is None:
            self.model['weights'] = weights.reshape((1,3))
            self.model['intercept'] = np.array(intercept)
            self.model['mu'] = np.array(mu)
        else:
            self.model['weights'] = np.append(self.model['weights'], weights.reshape((1,3)), axis=0)
            self.model['intercept'] = np.append(self.model['intercept'], intercept)
            self.model['mu'] = np.append(self.model['mu'], mu)


    """ Fit method  """
    def fit(self, X, y, f_0):

        alpha = self.compute_alpha(X)
        print('alpha',alpha.shape)
        """ Determine linear input layer """
        """ Eventually let user pass different classifiers in for this step """
        if f_0 is None:
            f_0, A, a = self.linear_reg(X, y)
        else: 
            f_0 = f_0

        """ Call algorithm """
        """ #### """
        i = 0
        while (i<=self.depth):
            print(i)
            ''' Initial layer assumes 1 for mu '''
            if i==0:
                mu = 1
                Xn = self.build_X1(f_0, alpha)

                # Number of rows
                m = Xn.shape[0]

                # Initial linear regression w/ mu equal to 1
                f_k, A, a = self.linear_reg(Xn, y)

                # Error metrics
                mse = self.mse(f_k, y, m)
                pred = np.where(f_k >= 0.5, 1, 0)

                print(y)
                acc = accuracy_score(y, pred)
                #mse = np.sum((f_k - y)**2) / Xn.shape[0]

                


            else:
                mu = self.minimize(f_k, A, a, alpha)
                Xn = self.build_Xn(f_k, A, a, alpha, mu)
                f_k, A, a = self.linear_reg(Xn, y)
                mse = self.mse(f_k, y, m)
                pred = np.where(f_k >= 0.5, 1, 0)
                acc = accuracy_score(y, pred)
                print(A.shape)
                #mse = np.sum((f_k - y)**2) / Xn.shape[0]
            '''
            # move MSE here, change function to receive logistic flag
            mse = self.error(f_k, y, m, logstic)'''
            
            # Save layer
            self.logging(A, a, mu)

            # add layers
            print('Iteration:', i, " Mu:", mu, "MSE:", mse, "Accuracy:", acc)

            i += 1
        """ #### """

    def minimize(self, f_k, A, a, alpha):
        print(f_k.shape)
        print(A.shape)
        print(a.shape)
        print(alpha.shape)
        self.f_k = f_k
        self.A = A
        self.alpha = alpha
        self.a = a
        res = minimize_scalar(self.f, bounds=(0,5000000000), method='bounded')
        return res.x
        

    def mse(self, f_k, y, m):
        return np.sum((f_k - y)**2) / m        

    def plot_error():
        pass

