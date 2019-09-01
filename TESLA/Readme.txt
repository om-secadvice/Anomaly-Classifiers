HOW TO USE submit.py
#####################
1. Put submit.py in same folder as dataset(csv or excel files only).
2. Execute
   python3 -W ignore submit.py [Dataset] [OutputColumn] [desired_model] [C] [n_top_features] [FeatureDropList]
3. Done!!!

ARGUEMENTS
-------------
1.-W ignore: Suppress python warnings.
2.Dataset: Complete Dataset Feature and labels in .csv or excel files. 
3.OutputColumn: Field name of output column inside dataset
4.desired_model:linear_svm,tree,logistic
5.C: C value for linear_svm or logistic Regression (Inverse strength of regularization) (Default:1.0)
6.n_top_features:- Integer specifies no. of most informative features. Every execution of submit.py creates a new file "dfs.xlsx" with only those top features. We can train a new model on "dfs.xls" (Default:10)
6.FeatureDropList: Field names of the features which should not be considered.

EXAMPLES
#############
1.	python3 -W ignore submit.py HW_TESLA.xls STATIC svm 4 6
2.	python3 -W ignore submit.py HW_TESLA.xls STATIC logistic 2 10 TESLA
3.	python3 -W ignore submit.py HW_TESLA.csv STATIC tree
4.	python3 -W ignore submit.py HW_TESLA.xls STATIC

BEST RESULT
#############
tree>svm>logistic
Note:- 1.Every Shuffle gives a different value of accuracy for all three models. Accuracy even reaches 99.9% for some shuffles.
       2.Try changing the C value to 4 to get good result in Linear SVM.

DEPENDENCIES
#############
python 3.x.x
pandas,xlrd,sklearn is essential
#cloudpickle is optional
