# Book-ratings
## Can we predict the rating score of sci-fi books from the dataset?

We consult the [sf_military.csv](https://github.com/mariaciko/Book-ratings/blob/main/sf_military.csv) dataset referenced in the [Jupyter Notebook](https://github.com/mariaciko/Book-ratings/blob/main/CikoM_Mac_A2.ipynb) for this project.

The question that drives this project requires implementation of machine learning tools, including pipelines (and transformers), training various models i.e.. KNN, Naive Bayes, LDA, QDA) and evaluating testing sets, cross-fold validation, interepreting confusion matrices, etc. 

In the second part, we use a support vector machine (SVM) classifier and experiment with different kernel types (i.e. linear, polynomial, and RBF or radial basis function) to compute accuracies and mean validation scores, as well as visualizing decision boundaries.

We then build a simple neural network using Keras (TensorFlow) and gauge the effect of increasing the number of hidden layers on accuracy and model performance.

Lastly, we build a Random Forest (RF) classifier which uses bootstrap sampling from multiple decision trees and averages results. We show that we can implement the AdaBoost method to achieve even higher accuracy at the price of overfitting. 
