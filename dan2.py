import numpy as np
import pickle
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from time import strftime, gmtime

class DAN2Regressor(object):

    def __init__(self, depth=10, bounds=(0,5000000000)):

        #if kwargs['model'] is None:
        self.model = {
            'name': strftime('dan2model-'+ str(depth) + '-%Y-%b-%d-%H-%M-%S', gmtime()),
            'weights': None,
            'intercept': None,
            'mu': 1,
            'f_0': None,
            'f_k': None,
            'A': None,
            'alpha': None,
            'a': None,
            'bounds': bounds,
            'depth': depth,
            'lr': None,
            'mu_hist': None
        }

        #else:
        #    self.model = kwargs['model']
            


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
        #print('R', R.shape)

        """ Compute dot product """
        X_dot_R = (1 + np.dot(X,R))
        #print('XdR', X_dot_R.shape)
        X_dot_R = X_dot_R.reshape((len(X),))
        #print('XdR', X_dot_R.shape)

        """ Compute X and R magnitudes """
        X_mag = np.sqrt(1*1 + np.sum(np.square(X), axis=1))
        R_mag = np.sqrt(np.sum(R**2) + 1*1)

        """ Compute arccosine """
        acos = np.arccos(X_dot_R / (X_mag * R_mag))
        #print('acos', acos.shape)

        return acos.reshape(len(acos),1) 


    """ Linear method """
    def linear_reg(self, X, y):
        self.model['lr'] = LinearRegression(fit_intercept=True).fit(X, y)
        return self.model['lr'].predict(X), self.model['lr'].coef_[0], self.model['lr'].intercept_


    ''' '''
    def build_X1(self, f, alpha):
        #print('f', f.shape)
        #print('alpha', alpha.shape)

        return np.hstack((f, np.cos(alpha*self.model['mu']), np.sin(alpha*self.model['mu'])))


    ''' '''
    def build_Xn(self, f, A, a, alpha, mu):
        rows = f.shape[0]
        return np.hstack((A[0]*f, A[1]*np.cos(alpha*mu), A[2]*np.sin(alpha*mu)))


    def logging(self, weights, intercept, mu):
        if self.model['weights'] is None:
            self.model['weights'] = weights.reshape((1,3))
            self.model['intercept'] = np.array(intercept)
            self.model['mu_hist'] = np.array(mu)
        else:
            self.model['weights'] = np.append(self.model['weights'], weights.reshape((1,3)), axis=0)
            self.model['intercept'] = np.append(self.model['intercept'], intercept)
            self.model['mu_hist'] = np.append(self.model['mu'], mu)


    """ Fit method  """
    def fit(self, X, y):

        ## Get non-linear projection of input records
        self.model['alpha'] = self.compute_alpha(X)
        
        ## Get linear model from n input cols
        self.model['f_0'], self.model['A'], self.model['a'] = self.linear_reg(X, y)
        

        """ Call algorithm """
        """ #### """
        i = 0
        while (i<=self.model['depth']):
            ''' Initial layer assumes 1 for mu '''
            if i==0:
                Xn = self.build_X1(self.model['f_0'], self.model['alpha'])

                # Number of rows
                m = Xn.shape[0]

                # Initial linear regression w/ mu equal to 1
                self.model['f_k'], self.model['A'], self.model['a'] = self.linear_reg(Xn, y)

                # Error metrics
                mse = self.mse(self.model['f_k'], y, m)
                pred = np.where(self.model['f_k'] >= 0.5, 1, 0)
                acc = accuracy_score(y, pred)
                #mse = np.sum((f_k - y)**2) / Xn.shape[0]

            else:
                self.model['mu'] = self.minimize(self.model['f_k'], self.model['A'], self.model['a'], self.model['alpha'])
                Xn = self.build_Xn(self.model['f_k'], self.model['A'], self.model['a'], self.model['alpha'], self.model['mu'])
                self.model['f_k'], self.model['A'], self.model['a'], = self.linear_reg(Xn, y)
                mse = self.mse(self.model['f_k'], y, m)
                pred = np.where(self.model['f_k'] >= 0.5, 1, 0)
                acc = accuracy_score(y, pred)
                #print(A.shape)
                #mse = np.sum((f_k - y)**2) / Xn.shape[0]
            '''
            # move MSE here, change function to receive logistic flag
            mse = self.error(f_k, y, m, logstic)'''
            
            # Save layer
            self.logging(self.model['A'], self.model['a'], self.model['mu'])

            # add layers
            print('Iteration:', i, " Mu:", self.model['mu'], "MSE:", mse, "Accuracy:", acc)

            i += 1
        """ #### """


    def minimize(self, f_k, A, a, alpha):
        #print(f_k.shape)
        #print(A.shape)
        #print(a.shape)
        #print(alpha.shape)
        self.f_k = f_k
        self.A = A
        self.alpha = alpha
        self.a = a
        res = minimize_scalar(self.f, bounds=self.model['bounds'], method='bounded')
        return res.x
        

    def mse(self, f_k, y, m):
        return np.sum((f_k - y)**2) / m        


    def plot_error():
        pass

    

