import numpy as np

class LogisticRegression():
    
    def __init__(self):
        self.X = None
        self.y = None
        self.features = None
        self.bias = None
        self.num_training_ex = None
        self.num_features = None

    def fit(self,X,y,learning_rate = 0.01, num_iters = 2000,print_cost = True):
        '''
        Args:
        X -- matrix of training examples, numpy array dim(num_features,num_training_ex)
        y -- vector of labels, numpy array dim(1,num_training_x)

        Returns:
        params -- dictionary containing the weights w and bias b
            keys:
                'w' -- weights, numpy array (num_features,1)
                'b' -- bias, scalar
        grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
            keys:
                'dw' -- gradient of loss function with respect to w,numpy array (num_features,1) 
                'db -- gradient of the loss function with respect to b, scalar
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
        '''
        # setting and initializing object parameters 
        self.X,self.y = X,y
        self._initialize_num_features_and_num_training_ex(self.X)
        self._initialize_with_zeros(self.num_features)
        dw,db = None, None
        costs = []

        # GRADIENT DESCENT
        for i in range(num_iters):

            # calculating grad and cost
            grads,cost = self._propagate(self.X,self.y,self.features,self.bias)

            # retrieving derivatives
            dw,db = grads['dw'], grads['db']

            # update rule
            self.features-= learning_rate*dw
            self.bias-=learning_rate*db

            # append to costs
            if i % 100 == 0:
                    costs.append(cost)

            # check if print cost is True
            if print_cost:
                # record cost every 100 steps
                if i%100 ==0:
                    print(f"Cost after iteration{i}: {cost}")
        
        params = {
            'w':self.features,
            'b':self.bias
        }

        grads = {
            'dw':dw,
            'db':db 
        }

        return params,grads,costs

    def predict(self,X):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression (parameters features and bias)
        
        Arguments:
        X -- data of size (num_features, number of examples)
        
        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        '''
        # retrieving and adjusting parameters
        m = X.shape[1]
        y_preds = np.zeros((1,m))
        w = self.features.reshape((self.num_features,1))
        b = self.bias

        # calculating the probabilities using sigmoid function
        probs = self._sigmoid(w.T.dot(X)+b)

        # predict 0 if <=0.5 else 1
        for i in range(probs.shape[1]):
            if probs[0,i]<=0.5:
                y_preds[0,i]=0
            else:
                y_preds[0,i]=1

        return y_preds

    def _propagate(self, X, y, w, b):
        '''
        Args:
        X -- data (input) matrix, numpy array (num_features,num_training_ex)
        y -- labels vector. numpy array (1,num_training_x)
        w -- weights, numpy array (num_features,1)
        b -- bias, scalar

        Return:
        grads -- dictionary with the gradients of the backwards propagation (dw,db)
            keys:
              dw -- gradient of the loss function with respect to w, numpy array (num_features,1)
              db -- gradient of the loss function with respect to b, scalar
        cost -- negative log likelihood cost for logistic regression
        '''
        m = self.num_training_ex
        # FORWARD PROPAGATION
        z = np.dot(w.T, X)+b
        A = self._sigmoid(z)
        cost = -1/m*(y.dot(np.log(A).T+(1-y).dot(np.log(1-A).T)))

        # BACKWARDS PROPAGATION
        dz = A-y
        dw = 1/m*(np.dot(X, dz.T))
        db = np.sum(dz)/m

        grads = {
            'dw': dw,
            'db': db
        }

        return grads, cost

    def _sigmoid(self,z):
        '''
        Args:
        z -- scalar or numpy array

        Returns:
         The mathematical sigmoid of z -- scalar or numpy array
        '''
        return 1/(1+np.exp(-z))

    def _initialize_with_zeros(self,dim):
        '''
        Initialize the features vector and the bias with zeros

        Args:
        dim -- num_of_features, scalar

        Returns:
        features -- matrix of zeros, numpy array (dim,1) 
        '''
        self.features = np.zeros((dim,1))
        self.bias = 0
        return {
            'features':self.features,
            'bias': self.bias
        }

    def _initialize_num_features_and_num_training_ex(self,X):
        self.num_training_ex = X.shape[1]
        self.num_features = X.shape[0]
