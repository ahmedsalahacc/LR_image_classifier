from linear_model import LogisticRegression
import matplotlib.pyplot as plt
from scipy import ndimage
from h5py import File
from PIL import Image
import numpy as np
import scipy

def load_dataset(train_path,test_path):
    # train data
    train_dataset = File(train_path,'r')
    X_train = np.array(train_dataset["train_set_x"][:])
    y_train = np.array(train_dataset["train_set_y"][:])
    # test data
    test_dataset = File(test_path, 'r')
    X_test = np.array(test_dataset["test_set_x"][:])
    y_test = np.array(test_dataset["test_set_y"][:])
    
    return X_train, y_train, X_test, y_test

def flatten_image_tensor(img_tensor):
    """
     Reshape the training and test data sets so that images of 
     size (num_px, num_px, 3) are flattened into single vectors of shape (num_px  ∗  num_px  ∗  3, 1).
    """
    img_matrix = img_tensor.reshape(img_tensor.shape[0], -1).T
    return img_matrix

def tensorify(img_matrix,num_px):
    return img_matrix.reshape(img_matrix[1],num_px,num_px,3)

def standardize(img_mat):
    return img_mat/255

def destandardize(img_mat):
    return img_mat*255

def visualize_image(index,image_list):
    plt.imshow(image_list[index])

def predict_custom_img(path,X_train,y_train,num_px,label_map):
    lr = LogisticRegression()
    lr.fit(X_train,y_train,print_cost=False)
    # adjsuting image
    image = np.array(ndimage.imread(path, flatten=False))
    standardized_image = image/255.
    custom_image_flattened_standardized = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((1, num_px*num_px*3)).T
    # predicting image
    label = lr.predict(custom_image_flattened_standardized)
    label = label[0][0]
    # visualizing image
    plt.imshow(image)
    print(f"Classifier predicts label {label} ({label_map[label]})")
