# Pandas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd
# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np
# Matplotlib is a plotting library for python and pyplot gives us a MatLab like plotting framework. We will use this in our plotter function to plot data.
import matplotlib.pyplot as plt
#Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics
import seaborn as sns
# Preprocessing allows us to standarsize our data
from sklearn import preprocessing
# Allows us to split our data into training and testing data
from sklearn.model_selection import train_test_split
# Allows us to test parameters of classification algorithms and find the best one
from sklearn.model_selection import GridSearchCV
# Logistic Regression classification algorithm
from sklearn.linear_model import LogisticRegression
# Support Vector Machine classification algorithm
from sklearn.svm import SVC
# Decision Tree classification algorithm
from sklearn.tree import DecisionTreeClassifier
# K Nearest Neighbors classification algorithm
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
##%matplotlib inline


# function for plotting the confussion matrix
def plot_confusion_matrix(y,y_predict,label):
    "this function plots the confusion matrix"
    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    title = 'Confusion Matrix '+ label
    ax.set_title(title); 
    ax.xaxis.set_ticklabels(['did not land', 'land']); ax.yaxis.set_ticklabels(['did not land', 'landed'])
    nameFig = 'ConfussionMaterix_Multivariate_RidgeRegression_'+label+'.png'
    plt.savefig(nameFig)
    #plt.show()
    plt.close()


data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv")

# If you were unable to complete the previous lab correctly you can uncomment and load this csv

# data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/api/dataset_part_2.csv')

data.head()

X = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv')

# If you were unable to complete the previous lab correctly you can uncomment and load this csv

# X = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/api/dataset_part_3.csv')

print(X.head(100))


y = data.Class.to_numpy()
print(y)

# students get this 
transform = preprocessing.StandardScaler()
#
# lets normalize eqch feature simulataneouly
from sklearn.preprocessing import StandardScaler
# lets create the object that will normalize our features in X
SCALE = StandardScaler()

SCALE.fit(X.loc[:,X.columns])
x_scaled = SCALE.transform(X.loc[:,X.columns])


# logistic regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


# For this method I previouly tested various ratios when I slipt the data from train_loan.csv. For each test I also checked several values of the
# crossvalidation parameter and again, for each of these test I did a grid-search for finding the optimal imput parameters for the Logistic Regression algorithm.
# I comented the steps for finding the best test size and crossvalidation values and I am just runnig the case that has the optimal values of such parameters

print("\n\n ******************** Logistic Regression *******************")
from sklearn.linear_model import Ridge

parameters ={"C":[0.01,0.1,1],'penalty':['l2'], 'solver':['lbfgs']}# l1 lasso l2 ridge
lr=LogisticRegression()
#
best_scheme_lr = np.zeros(8) # R^2 sore, K values, cv value
best_params_lr = []


for test_size_n in range(3,11):
#for test_size_n in range(10,11):
    print("============== Test Sice: 1/",test_size_n)
    x_train_lr, x_test_lr, y_train_lr, y_test_lr = train_test_split( X, y, test_size=float(1/test_size_n), random_state=4)
    #
    #for cvn in range(3,11):
    for cvn in range(3,4):
        print("--------- Cross Validation cv: ",cvn)

        logreg_cv = GridSearchCV(lr,parameters,cv = cvn,return_train_score=True) # logreg_cv ???
        print("\n GridSearchCV")
        print(logreg_cv)
        #
        # lets fit the model to the training targets
        logreg_cv.fit(x_train_lr,y_train_lr)
        # lets fit the model to the training targets
        logreg_cv.fit(x_train_lr,y_train_lr)
        print("\n Estimalor List")
        print(logreg_cv.estimator)
        #lets find the best values for the free parameters
        print("\n lets find the best values for the free parameters")
        print(logreg_cv.best_estimator_)
        #
        # lets get the mean sccore in the cross-validation data
        scores = logreg_cv.cv_results_ #The resulting score of the different free parameters are stored in this dictionary
        ##print("\n lets get the mean sccore in the cross-validation data")
        ##print(scores['mean_test_score'])
        #
        # lets print out the score for the different free parameter values
        #print(scores)
        ##print("\n lets print out the score for the different free parameter values")
        ##for param, mean_val, mean_train in zip( scores['params'],scores['mean_test_score'],scores['mean_train_score'] ):
        ##    print(param,"R^2 on test data", mean_val,"R^2 on train data", mean_train)
        #
        print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
        print("accuracy :",logreg_cv.best_score_)

        print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
        print("accuracy with training data :",logreg_cv.best_score_)
        accuracy_lr = logreg_cv.best_score_
        print("accuracy :",accuracy_lr)
        #
        #
        yhat_lr_params =logreg_cv.predict(x_test_lr)
        yhat_train_lr_params = logreg_cv.predict(x_train_lr)
        #
        print("\n Lets calculate the R^2 for the training data set")
        trainingScore_lr = logreg_cv.score(x_train_lr,y_train_lr)
        print(trainingScore_lr)
        print("\n Lets calculate the R^2 for the testing data set")
        testScore_lr = logreg_cv.score(x_test_lr,y_test_lr)
        print(testScore_lr)
        #
        #from sklearn.metrics import jaccard_score
        jaccardVal_lr = jaccard_score(y_test_lr, yhat_lr_params)
        print("jaccard_score: ",jaccardVal_lr)
        #from sklearn.metrics import f1_score
        f1Val_lr = f1_score(y_test_lr, yhat_lr_params, average='weighted')
        print("f1_score: ",f1Val_lr)
        #from sklearn.metrics import log_loss
        loglossVal_lr = log_loss(y_test_lr, yhat_lr_params)
        print("log_loss score: ",loglossVal_lr)
        #
        #if (best_scheme_lr[4]<accuracy_lr):
        if (best_scheme_lr[4]<f1Val_lr):
            best_scheme_lr = [accuracy_lr,trainingScore_lr,testScore_lr,jaccardVal_lr,f1Val_lr,loglossVal_lr,test_size_n,cvn]
            best_params_lr = logreg_cv.best_estimator_
################################################## Plotting confussin matrix
        label = "logisticRegression_params_test_size_"+str(test_size_n)+"_crossVal_"+str(cvn)
        plot_confusion_matrix(y_test_lr,yhat_lr_params,label)
        #
print("************************** GRID SEARCH RESULTS")
print("Logistic Regression Best score")
print(best_scheme_lr)
print("Logistic Regression Best parameters")
print(best_params_lr)
"""
tuned hpyerparameters :(best parameters)  {'C': 0.1, 'penalty': 'l2', 'solver': 'lbfgs'}
accuracy : 0.8196428571428571

 Lets calculate the R^2 for the testing data set
0.8333333333333334

Criteria: Jaccard
Logistic Regression Best score
[0.8024691358024691, 0.8888888888888888, 0.8888888888888888, 0.8571428571428571, 0.882051282051282, 3.837730665815654, 10, 3]
Logistic Regression Best parameters
LogisticRegression(C=1)
tuned hpyerparameters :(best parameters)  {'C': 1, 'penalty': 'l2', 'solver': 'lbfgs'}

Criteria: f1_score
Logistic Regression Best score
[0.8024691358024691, 0.8888888888888888, 0.8888888888888888, 0.8571428571428571, 0.882051282051282, 3.837730665815654, 10, 3]
Logistic Regression Best parameters
LogisticRegression(C=1)

Criteria: accuracy
Logistic Regression Best score
[0.8024691358024691, 0.8888888888888888, 0.8888888888888888, 0.8571428571428571, 0.882051282051282, 3.837730665815654, 10, 3]
Logistic Regression Best parameters
LogisticRegression(C=1)
"""




# For this method I previouly tested various ratios when I slipt the data from train_loan.csv. For each test I also checked several values of the
# crossvalidation parameter and again, for each of these test I did a grid-search for finding the optimal imput parameters for the Support Vector Machine algorithm.
# I comented the steps for finding the best test size and crossvalidation values and I am just runnig the case that has the optimal values of such parameters

print("\n\n ******************** SUPPORT VECTOR MACHINE *******************")
from sklearn.svm import SVC
parameters_svm = {'kernel':('linear', 'rbf','poly','rbf', 'sigmoid'),
              'C': np.logspace(-3, 3, 5),
              'gamma':np.logspace(-3, 3, 5)}
svm = SVC()
#
best_scheme_svm = np.zeros(7) # R^2 sore, K values, cv value
best_params_svm = []


for test_size_n in range(3,11):
#for test_size_n in range(10,11):
    print("============== Test Sice: 1/",test_size_n)
    x_train_svm, x_test_svm, y_train_svm, y_test_svm = train_test_split( X, y, test_size=float(1/test_size_n), random_state=4)
    #
    #for cvn in range(3,11):
    for cvn in range(7,8):
        print("--------- Cross Validation cv: ",cvn)
        svmreg_cv = GridSearchCV(svm,{'kernel':('rbf', 'sigmoid'),'C': np.logspace(-3, 3, 5),'gamma':np.logspace(-3, 3, 5)},cv = cvn,return_train_score=True)
        print("\n GridSearchCV")
        print(svmreg_cv)
        # lets fit the model to the training targets
        svmreg_cv.fit(x_train_svm,y_train_svm)
        print("\n Estimalor List")
        print(svmreg_cv.estimator)
        #lets find the best values for the free parameters
        print("\n lets find the best values for the free parameters")
        print(svmreg_cv.best_estimator_)
        #
        #
        # lets get the mean sccore in the cross-validation data
        scores = svmreg_cv.cv_results_ #The resulting score of the different free parameters are stored in this dictionary
        ##print("\n lets get the mean sccore in the cross-validation data")
        ##print(scores['mean_test_score'])
        #
        ### lets print out the score for the different free parameter values
        ###print(scores)
        print("\n lets print out the score for the different free parameter values")
        ##for param, mean_val, mean_train in zip( scores['params'],scores['mean_test_score'],scores['mean_train_score'] ):
        ##    print(param,"R^2 on test data", mean_val,"R^2 on train data", mean_train)
        #
        print("tuned hpyerparameters :(best parameters) ",svmreg_cv.best_params_)
        print("accuracy with training data :",svmreg_cv.best_score_)
        accuracy_svm = svmreg_cv.best_score_
        print("accuracy :",accuracy_svm)
        #
        #
        yhat_svm_params =svmreg_cv.predict(x_test_svm)
        yhat_train_svm_params = svmreg_cv.predict(x_train_svm)
        #
        print("\n Lets calculate the R^2 for the training data set")
        trainingScore_svm = svmreg_cv.score(x_train_svm,y_train_svm)
        print(trainingScore_svm)
        print("\n Lets calculate the R^2 for the testing data set")
        testScore_svm = svmreg_cv.score(x_test_svm,y_test_svm)
        print(testScore_svm)
        #
        #from sklearn.metrics import jaccard_score
        #print("jaccard_score: ",jaccard_score(y_test, yhat_svm_params,pos_label=2))
        #from sklearn.metrics import jaccard_score
        jaccardVal_svm = jaccard_score(y_test_svm, yhat_svm_params)
        print("jaccard_score: ",jaccardVal_svm)
        #from sklearn.metrics import f1_score
        f1Val_svm = f1_score(y_test_svm, yhat_svm_params, average='weighted')
        print("f1_score: ",f1Val_svm)
        #from sklearn.metrics import log_loss
        ##print("log_loss score: ",log_loss(y_test, yhat_svm_params))
        #
        #if (best_scheme_svm[4]<accuracy_svm):
        if (best_scheme_svm[4]<f1Val_svm):
            best_scheme_svm = [accuracy_svm,trainingScore_svm,testScore_svm,jaccardVal_svm,f1Val_svm,test_size_n,cvn]
            best_params_svm = svmreg_cv.best_estimator_
################################################## Plotting confussin matrix
        label = "svm_params_test_size_"+str(test_size_n)+"_crossVal_"+str(cvn)
        plot_confusion_matrix(y_test_svm,yhat_svm_params,label)

print("************************** GRID SEARCH RESULTS")
print("SUPPORT VECTOR MACHINE Best score")
print(best_scheme_svm)
print("SUPPORT VECTOR MACHINE Best parameters")
print(best_params_svm)
"""
tuned hpyerparameters :(best parameters)  {'C': 0.001, 'gamma': 0.001, 'kernel': 'rbf'}
accuracy with testing data : 0.6678571428571429

 Lets calculate the R^2 for the testing data set
0.6666666666666666

Criteria: jaccard
SUPPORT VECTOR MACHINE Best score
[0.7202380952380952, 0.9666666666666667, 0.5666666666666667, 0.5666666666666667, 0.4822695035460993, 3, 7]
SUPPORT VECTOR MACHINE Best parameters
SVC(gamma=0.001)
tuned hpyerparameters :(best parameters)  {'C': 1.0, 'gamma': 0.001, 'kernel': 'rbf'}


criteria: f1_score
SUPPORT VECTOR MACHINE Best score
[0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.5333333333333334, 10, 7]
SUPPORT VECTOR MACHINE Best parameters
SVC(C=0.001, gamma=0.001)
tuned hpyerparameters :(best parameters)  {'C': 0.001, 'gamma': 0.001, 'kernel': 'rbf'}

criteria: accuracy
SUPPORT VECTOR MACHINE Best score
[0.7202380952380952, 0.9666666666666667, 0.5666666666666667, 0.5666666666666667, 0.4822695035460993, 3, 7]
SUPPORT VECTOR MACHINE Best parameters
SVC(gamma=0.001)
"""


# For this method I previouly tested various ratios when I slipt the data from train_loan.csv. For each test I also checked several values of the
# crossvalidation parameter and again, for each of these test I did a grid-search for finding the optimal imput parameters for the Decision Tree algorithm.
# I comented the steps for finding the best test size and crossvalidation values and I am just runnig the case that has the optimal values of such parameters

print("\n\n ******************** DECISION TREE *******************")


from sklearn.tree import DecisionTreeClassifier

parameters = {'criterion': ['gini', 'entropy'],
     'splitter': ['best', 'random'],
     'max_depth': [2*n for n in range(1,10)],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 14]}
#
tree = DecisionTreeClassifier()
#
best_scheme_tree = np.zeros(7) # R^2 sore, K values, cv value
best_params_tree = []

for test_size_n in range(3,11):
#for test_size_n in range(10,11):
    print("============== Test Sice: 1/",test_size_n)
    x_train_tree, x_test_tree, y_train_tree, y_test_tree = train_test_split( X, y, test_size=float(1/test_size_n), random_state=4)
    #
    #for cvn in range(3,11):
    for cvn in range(3,4):
        print("--------- Cross Validation cv: ",cvn)
        treereg_cv = GridSearchCV(tree,parameters,cv = cvn,return_train_score=True)
        treereg_cv.fit(x_train_tree,y_train_tree)
        #
        print("\n Estimalor List")
        print(treereg_cv.estimator)
        #lets find the best values for the free parameters
        print("\n lets find the best values for the free parameters")
        print(treereg_cv.best_estimator_)
        #
        # lets get the mean sccore in the cross-validation data
        scores = treereg_cv.cv_results_ #The resulting score of the different free parameters are stored in this dictionary
        ##print("\n lets get the mean sccore in the cross-validation data")
        ##print(scores['mean_test_score'])
        #
        ### lets print out the score for the different free parameter values
        ###print(scores)
        ##print("\n lets print out the score for the different free parameter values")
        ##for param, mean_val, mean_train in zip( scores['params'],scores['mean_test_score'],scores['mean_train_score'] ):
        ##    print(param,"R^2 on test data", mean_val,"R^2 on train data", mean_train)
        #
        #
        print("tuned hpyerparameters :(best parameters) ",treereg_cv.best_params_)
        accuracy_tree = treereg_cv.best_score_
        print("accuracy :",accuracy_tree)
        #
        #
        yhat_tree_params =treereg_cv.predict(x_test_tree)
        yhat_train_tree_params = treereg_cv.predict(x_train_tree)
        #
        print("\n Lets calculate the R^2 for the training data set")
        trainingScore_tree = treereg_cv.score(x_train_tree,y_train_tree)
        print(trainingScore_tree)
        print("\n Lets calculate the R^2 for the testing data set")
        testScore_tree = treereg_cv.score(x_test_tree,y_test_tree)
        print(testScore_tree)
        #
        #from sklearn.metrics import jaccard_score
        #print("jaccard_score: ",jaccard_score(y_test, yhat_tree_params,pos_label=2))
        jaccardVal_tree = jaccard_score(y_test_tree, yhat_tree_params)
        print("jaccard_score: ",jaccardVal_tree)
        #from sklearn.metrics import f1_score
        f1Val_tree = f1_score(y_test_tree, yhat_tree_params, average='weighted')
        print("f1_score: ",f1Val_tree)
        #from sklearn.metrics import log_loss
        ##print("log_loss score: ",log_loss(y_test, yhat_tree_params))
        #
        #if (best_scheme_tree[4]<accuracy_tree):
        if (best_scheme_tree[4]<f1Val_tree):
            best_scheme_tree = [accuracy_tree,trainingScore_tree,testScore_tree,jaccardVal_tree,f1Val_tree,test_size_n,cvn]
            best_params_tree = treereg_cv.best_estimator_
################################################## Plotting confussin matrix
        label = "tree_params_test_size_"+str(test_size_n)+"_crossVal_"+str(cvn)
        plot_confusion_matrix(y_test_tree,yhat_tree_params,label)


print("************************** GRID SEARCH RESULTS")
print("DECISION TREE Best score")
print(best_scheme_tree)
print("DECISION TREE Best parameters")
print(best_params_tree)

"""
tuned hpyerparameters :(best parameters)  {'criterion': 'entropy', 'max_depth': 8, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'splitter': 'random'}
accuracy : 0.8892857142857145

 Lets calculate the R^2 for the training data set
0.9166666666666666

 Lets calculate the R^2 for the testing data set
0.9166666666666666

Criteria: jaccard
DECISION TREE Best score
[0.9102564102564102, 0.9230769230769231, 0.8333333333333334, 0.7777777777777778, 0.8229166666666666, 8, 3]
DECISION TREE Best parameters
DecisionTreeClassifier(max_depth=6, max_features='auto', min_samples_split=5)
tuned hpyerparameters :(best parameters)  {'criterion': 'gini', 'max_depth': 6, 'max_features': 'auto', 'min_samples_leaf': 4, 'min_samples_split': 10, 'splitter': 'random'}

Criteria: f1_socre
DECISION TREE Best score
[0.8888888888888888, 0.7777777777777778, 0.8888888888888888, 0.8571428571428571, 0.882051282051282, 10, 3]
DECISION TREE Best parameters
DecisionTreeClassifier(criterion='entropy', max_depth=2, max_features='auto',
                       min_samples_split=5)

tuned hpyerparameters :(best parameters)  {'criterion': 'entropy', 'max_depth': 2, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 5, 'splitter': 'best'}


Criteria: accuracy
DECISION TREE Best score
[0.9, 0.85, 0.8333333333333334, 0.8, 0.8148148148148148, 3, 3]
DECISION TREE Best parameters
DecisionTreeClassifier(max_depth=2, max_features='auto', min_samples_split=10,
                       splitter='random')
tuned hpyerparameters :(best parameters)  {'criterion': 'gini', 'max_depth': 2, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 10, 'splitter': 'random'}

"""



# For this method I previouly tested various ratios when I slipt the data from train_loan.csv. For each test I also checked several values of the
# crossvalidation parameter and again, for each of these test I did a grid-search for finding the optimal imput parameters for the k-neighbours algorithm.
# I comented the steps for finding the best test size and crossvalidation values and I am just runnig the case that has the optimal values of such parameters

print("\n\n ******************** K-neighbours *******************")

from sklearn.neighbors import KNeighborsClassifier
    
parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 14],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]}
KNN = KNeighborsClassifier()
#
best_scheme_knn = np.zeros(7) # R^2 sore, K values, cv value
best_params_knn = []

for test_size_n in range(3,11):
#for test_size_n in range(10,11):
    print("============== Test Sice: 1/",test_size_n)
    x_train_knn, x_test_knn, y_train_knn, y_test_knn = train_test_split( X, y, test_size=float(1/test_size_n), random_state=4)
    #
    #for cvn in range(3,11):
    for cvn in range(7,8):
        print("--------- Cross Validation cv: ",cvn)

        knn_reg_cv = GridSearchCV(KNN,parameters,cv = cvn,return_train_score=True) # knn_reg_cv ???
        print("\n GridSearchCV")
        print(knn_reg_cv)
        #
        # lets fit the model to the training targets
        knn_reg_cv.fit(x_train_knn,y_train_knn)
        # lets fit the model to the training targets
        knn_reg_cv.fit(x_train_knn,y_train_knn)
        print("\n Estimalor List")
        print(knn_reg_cv.estimator)
        #lets find the best values for the free parameters
        print("\n lets find the best values for the free parameters")
        print(knn_reg_cv.best_estimator_)
        #
        # lets get the mean sccore in the cross-validation data
        scores = knn_reg_cv.cv_results_ #The resulting score of the different free parameters are stored in this dictionary
        ##print("\n lets get the mean sccore in the cross-validation data")
        ##print(scores['mean_test_score'])
        #
        # lets print out the score for the different free parameter values
        #print(scores)
        ##print("\n lets print out the score for the different free parameter values")
        ##for param, mean_val, mean_train in zip( scores['params'],scores['mean_test_score'],scores['mean_train_score'] ):
        ##    print(param,"R^2 on test data", mean_val,"R^2 on train data", mean_train)
        #
        print("tuned hpyerparameters :(best parameters) ",knn_reg_cv.best_params_)
        print("accuracy :",knn_reg_cv.best_score_)

        print("tuned hpyerparameters :(best parameters) ",knn_reg_cv.best_params_)
        print("accuracy with training data :",knn_reg_cv.best_score_)
        accuracy_knn = knn_reg_cv.best_score_
        print("accuracy :",accuracy_knn)
        #
        #
        yhat_knn_params =knn_reg_cv.predict(x_test_knn)
        yhat_train_knn_params = knn_reg_cv.predict(x_train_knn)
        #
        print("\n Lets calculate the R^2 for the training data set")
        trainingScore_knn = knn_reg_cv.score(x_train_knn,y_train_knn)
        print(trainingScore_knn)
        print("\n Lets calculate the R^2 for the testing data set")
        testScore_knn = knn_reg_cv.score(x_test_knn,y_test_knn)
        print(testScore_knn)
        #
        #from sklearn.metrics import jaccard_score
        #print("jaccard_score: ",jaccard_score(y_test, yhat_knn_params,pos_label=2))
        jaccardVal_knn = jaccard_score(y_test_knn, yhat_knn_params)
        print("jaccard_score: ",jaccardVal_knn)
        #from sklearn.metrics import f1_score
        f1Val_knn = f1_score(y_test_knn, yhat_knn_params, average='weighted')
        print("f1_score: ",f1Val_knn)
        #from sklearn.metrics import log_loss
        ##print("log_loss score: ",log_loss(y_test, yhat_knn_params))
        #
        #if (best_scheme_knn[4]<accuracy_knn):
        if (best_scheme_knn[4]<f1Val_knn):
            best_scheme_knn = [accuracy_knn,trainingScore_knn,testScore_knn,jaccardVal_knn,f1Val_knn,test_size_n,cvn]
            best_params_knn = knn_reg_cv.best_estimator_
################################################## Plotting updatedd confussin matrix
            label = "knn_params_test_size_"+str(test_size_n)+"_crossVal_"+str(cvn)
            plot_confusion_matrix(y_test_knn,yhat_knn_params,label)

print("************************** GRID SEARCH RESULTS")
print("K nearest neighbours Best score")
print(best_scheme_knn)
print("K nearest neighbours Best parameters")
print(best_params_knn)

"""
tuned hpyerparameters :(best parameters)  {'algorithm': 'auto', 'n_neighbors': 3, 'p': 1}
accuracy : 0.6642857142857143

 Lets calculate the R^2 for the training data set
0.6111111111111112

 Lets calculate the R^2 for the testing data set
0.6111111111111112

Criteria: jaccard
K nearest neighbours Best score
[0.6655844155844156, 0.6790123456790124, 0.4444444444444444, 0.375, 0.4588744588744588, 10, 7]
K nearest neighbours Best parameters
KNeighborsClassifier(n_neighbors=10, p=1)
tuned hpyerparameters :(best parameters)  {'algorithm': 'auto', 'n_neighbors': 10, 'p': 1}

Criteria: f1_score
[0.6883116883116883, 0.8051948051948052, 0.5384615384615384, 0.45454545454545453, 0.5211538461538463, 7, 7]
K nearest neighbours Best parameters
KNeighborsClassifier(n_neighbors=4, p=1)
tuned hpyerparameters :(best parameters)  {'algorithm': 'auto', 'n_neighbors': 4, 'p': 1}

Criteria: accuracy
K nearest neighbours Best score
[0.7480158730158729, 0.8666666666666667, 0.5, 0.4642857142857143, 0.49293966623876767, 3, 7]
K nearest neighbours Best parameters
KNeighborsClassifier(n_neighbors=3, p=1)
tuned hpyerparameters :(best parameters)  {'algorithm': 'auto', 'n_neighbors': 3, 'p': 1}

"""