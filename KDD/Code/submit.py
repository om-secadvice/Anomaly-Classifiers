import sys
import pandas as pd
import pickle as pkl
import numpy as np
import itertools
from math import ceil 
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split,cross_val_score,ShuffleSplit
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  confusion_matrix
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
LabelY=41
errmsg='USAGE: learn.py DesiredModel C'

argl=len(sys.argv)
if(argl>3):
    print(errmsg)
    exit(1)

desired_model='forest'
C=1.0


if(argl>=2):
    desired_model=sys.argv[1]

if(argl==3):
    C=float(sys.argv[2])

def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)
    

le=pkl.load(open('label_encoders.pkl','rb'))

try:
    X_train=pkl.load(open('X_train.pkl','rb'))
    X_cv=pkl.load(open('X_cv.pkl','rb'))
    y_train=pkl.load(open('y_train.pkl','rb'))
    y_cv=pkl.load(open('y_cv.pkl','rb'))
    scaler=pkl.load(open('scaler.pkl','rb'))
    print('Pickles loaded...')
except:
    #Reading into pandas dataframe from excel file.
    print('Dataset loading...')
    df_train=pd.read_csv('kddtrain.csv',header=None)
    df_train=df_train.drop_duplicates()


    df=df_train

    sz,n_feat=df.shape
    print("Dataset Read, Size=",sz)
    print('Number of features=',n_feat-1)


    #Encoding Label Column to 0,1 for different classes say "Normal" or "Stressed". 

    df[LabelY]=le[41].transform(df[LabelY])

    
    encode_text_dummy(df, 1)
    df[2]=le[2].transform(df[2])
    encode_text_dummy(df, 3) 

    X=df.drop(LabelY,axis=1)
    y=df[LabelY]
   
    X_train, X_cv, y_train, y_cv = train_test_split(X,y,test_size=0.50,random_state=40,stratify=y)



    #Centering data by mean i.e. subtracting mean from each data
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_cv=scaler.transform(X_cv)

    pkl.dump(X_train,open('X_train.pkl','wb'))
    pkl.dump(X_cv,open('X_cv.pkl','wb'))
    pkl.dump(y_train,open('y_train.pkl','wb'))
    pkl.dump(y_cv,open('y_cv.pkl','wb'))
    pkl.dump(scaler,open('scaler.pkl','wb'))

models_list={'linear_svm':SVC(kernel='linear',C=C,class_weight='balanced'),
             'tree':DecisionTreeClassifier(class_weight='balanced'),
             'logistic':LogisticRegression(multi_class='multinomial', solver='newton-cg',class_weight='balanced',C=C),
             'forest':RandomForestClassifier(class_weight='balanced',n_estimators=200),   
             'BNB':BernoulliNB(),
             'Extree':ExtraTreeClassifier(),
             'Extrees':ExtraTreesClassifier(),
             'GNB':GaussianNB(),
             'KNN':KNeighborsClassifier(),
             'LinearSVC':LinearSVC(class_weight='balanced',multi_class='crammer_singer'),
             'MLP':MLPClassifier(solver='sgd'),
             'LDA':LinearDiscriminantAnalysis()}




clf=models_list[desired_model]
print('\n\nTraining ',desired_model,' model...')

clf.fit(X_train,y_train)

print('Model Trained')

print("Validation Accuracy=" ,clf.score(X_cv,y_cv))

df_test=pd.read_csv('test.csv',header=None)
df_test=df_test.drop_duplicates()

df_test[41]=le[41].transform(df_test[41])
encode_text_dummy(df_test, 1)
df_test[2]=le[2].transform(df_test[2])
encode_text_dummy(df_test, 3) 

X_test=scaler.transform(df_test.drop(41,axis=1))


y_test=df_test[41]


y_pred=clf.predict(X_test)


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)

print(cnf_matrix)
print("Testing Accuracy= ", clf.score(X_test,y_test)*100)
print("Classification Report:-\n",classification_report(y_test,y_pred))


