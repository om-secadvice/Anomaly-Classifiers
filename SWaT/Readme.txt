Secure Water Treatment Testbed Dataset
HOW TO USE cyber4.py
#####################
1. Put cyber4.py in same folder as dataset(csv or excel files only).
2. Execute
   python3 -W ignore cyber4.py [desired_model] [C]
3. Done!!!

ARGUMENTS
-------------
1.-W ignore: Suppress python warnings.
2.Dataset: Complete Dataset Feature and labels in .csv or excel files. 
3.desired_model:linear_svm,tree,logistic,forest etc
4.C: C value for linear_svm or logistic Regression (Inverse strength of regularization) (Default:3.0)
EXAMPLES
#############
1.	python3 -W ignore cyber4.py linear_svm 4
2.	python3 -W ignore cyber4.py logistic 2
3.	python3 -W ignore cyber4.py tree
4.	python3 -W ignore cyber4.py forest
5.	python3 -W ignore cyber4.py LDA

BEST RESULT
#############
Tree-based classifiers
Note:-Every Shuffle can give a different value of accuracy for all models. Accuracy even reaches 100.0%.

DEPENDENCIES
#############
python 3.x.x
pandas,xlrd,sklearn is essential
