from linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from img_utils import load_dataset
import numpy as np

def lr_model(X_train,y_train,X_test,y_test,learning_rate = 0.01, num_iters = 2000, print_cost = False):
    """
    Builds the logistic regression model
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    # insitantiating logistic regression object
    lr_classifier = LogisticRegression()

    # fitting model
    params, grads, costs = lr_classifier.fit(X_train,y_train,learning_rate,num_iters,print_cost)

    # predicting train/test set examples
    y_test_preds = lr_classifier.predict(X_test)
    y_train_preds = lr_classifier.predict(X_train)

    print("train accuracy: {} %".format(
        100 - np.mean(np.abs(y_train_preds - y_train)) * 100))
    print("test accuracy: {} %".format(
        100 - np.mean(np.abs(y_test_preds - y_test)) * 100))

    result = {
        'costs':costs,
        'y_test_preds':y_test_preds,
        'y_train_preds':y_train_preds,
        'weights':params['w'],
        'bias':params['b'],
        'learning_rate': learning_rate,
        'num_iters': num_iters
    }

    return result
