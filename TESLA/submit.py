import sys
import pandas as pd
import pickle as pkl
import numpy as np
from math import ceil
from sklearn.model_selection import train_test_split, cross_val_score,ShuffleSplit
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import  confusion_matrix
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.utils import shuffle
errmsg='USAGE: learn.py Dataset.xls LabelY DesiredModel top_features C [DropLabel1 DropLabel2...]\nOr model name incorrect'
models_list=['linear_svm','tree','logistic']
if(len(sys.argv)<3):
    print(errmsg)
    exit(1)

Dataset=sys.argv[1]
LabelY=sys.argv[2]
desired_model='linear_svm'
C=1.0
top_features=6
dropLabels=[]


if(len(sys.argv)>=4 and sys.argv[3] in models_list):
    desired_model=sys.argv[3]
else:
    print(errmsg)
    exit(1)

if(len(sys.argv)>=5):
    try:
        top_features=int(sys.argv[4])
    except:
        print('"top_features" should be integer.')

if(len(sys.argv)>=6):
    C=float(sys.argv[5])
    
if(len(sys.argv)>=7):
    dropLabels=sys.argv[6:]
    if(LabelY in dropLabels):
        print(LabelY,' cannot be in the list of labels to drop.')
        exit(2)



#Reading into pandas dataframe from excel file.
if(Dataset.find('xls')>=0):
    df=pd.read_excel(Dataset,dtype='unicode',skipinitialspace='true')


#Removing columns from dataset using droplabels list from cmdline(if any)  
if(dropLabels.__len__!=0):
        df=df.drop(dropLabels,axis=1)

df=df.drop_duplicates()
sz,n_feat=df.shape
print("Dataset Read, Size=",sz)
print('Number of features=',n_feat-1)

#Encoding Label Column to 0,1 for different classes say "Normal" or "Stressed". 
df[LabelY]=LabelEncoder().fit_transform(df[LabelY])

dfs=shuffle(df)

#Splitting into first 25% and last 75% for testing and training and cross val.
df_test=dfs.tail(ceil(sz*0.25))
df_train=dfs.head(int(sz*0.75))

#Separating feature matrix and corresponding labels
X_train=df_train.drop(LabelY,axis=1)
X_test=df_test.drop(LabelY,axis=1)
y_train=df_train[LabelY]
y_test=df_test[LabelY]

#Centering data by mean i.e. subtracting mean from each data
scaler=StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

   

models_list={'linear_svm':SVC(kernel='linear',C=C),'tree':DecisionTreeClassifier(),'logistic':LogisticRegression(C=C)}
clf=models_list[desired_model]
print('\n\nTraining ',desired_model,' model...')

clf.fit(X_train,y_train)

print('\nMost Important Features')
if(desired_model=='tree'):
    gini=np.asarray(clf.feature_importances_)
    top_gini=np.argsort(-gini)[:top_features]
    new_dfs=dfs[dfs.drop(LabelY,axis=1).columns[top_gini]]
    new_dfs[LabelY]=dfs[LabelY]
    new_dfs.to_excel('dfs.xlsx',index=False)
    print(list(new_dfs))
    print('Gini index\n',list(gini[top_gini]))

elif(desired_model=='linear_svm'):
    coef = np.absolute(clf.coef_.ravel())
    top_coefficients = np.argsort(-coef)[:top_features]
    new_dfs=dfs[dfs.drop(LabelY,axis=1).columns[top_coefficients]]
    new_dfs[LabelY]=dfs[LabelY]
    new_dfs.to_excel('dfs.xlsx',index=False)
    print(list(new_dfs))
    print('Absolute Values of Coeffs\n',list(coef[top_coefficients]))
    
elif(desired_model=='logistic'):
    coef = np.absolute(clf.coef_.ravel())
    top_coefficients = np.argsort(-coef)[:top_features]
    new_dfs=dfs[dfs.drop(LabelY,axis=1).columns[top_coefficients].ravel()]
    new_dfs[LabelY]=dfs[LabelY]
    new_dfs.to_excel('dfs.xlsx',index=False)
    print(list(new_dfs))
    print('Absolute Values of Coeffs\n',list(coef[top_coefficients]))


print("\nSaved file dfs.xlsx with selected features. Try Training on that file to see the quality of selected features.\nCross Validating...")


#Training on 50% of original data and 25% for cross validation.
#5 different splits of 25%,50% and taking average on validation score with some stdev
#This way we can effectively use all 75%  for training.(Leave One Out CV Method below)
cv = ShuffleSplit(n_splits=5, test_size=int(0.25*sz), random_state=0)
scores=cross_val_score(clf,X_train,y_train,cv=cv)
print("Validation Accuracy=%.4f Std(+/- %0.4f)" % (scores.mean()*100,scores.std()*200))

y_pred=clf.predict(X_test)


conf_mat=confusion_matrix(y_test,y_pred)
print("\n\nFalse Positive:",conf_mat[0][1])
print("False Negative:",conf_mat[1][0])
print("Testing Accuracy= ", clf.score(X_test,y_test)*100)

