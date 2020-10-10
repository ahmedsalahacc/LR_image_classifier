# LR_image_classifier
Logistic Regression image classifier fully implemented from scratch. I use the classifier example to recognize cats. By changing the dataset and following the pattern, any image dataset can be used in the classifier.

## Main flow
* The "datasets" directory contains image datasets stored in the .h5 format
*'LR classifier.ipynb' notebook contains the main flow of the classifier.
* img_utils contains the neccessary tools and utilities to format, standardize, flatten, and predict custom images
* linear_model.py contains the LogisticRegression class
* models.py contains the model that is used to train and test the dataset
* "images" directory contains custom images that can be used as a standalone prediction

If you wish to predict your custom image, make sure to add your image to the images directory (for the sake of the pattern), and copy its path into the predict function at the end of the 'LR classifier.ipynb' notebook
