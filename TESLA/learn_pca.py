import sys
import pandas as pd
import pickle as pkl
from math import floor
from sklearn.model_selection import train_test_split, cross_val_score,ShuffleSplit
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
if(len(sys.argv)<3):
    print('learn.py Dataset.csv/.xlsx LabelY Classifier DegreeRbf [DropLabel1 DropLabel2...]')
    exit(1)

Dataset=sys.argv[1]
LabelY=sys.argv[2]
desired_model='svm'
degree_rbf=1
dropLabels=[]
if(len(sys.argv)>=4):
    desired_model=sys.argv[3]
if(len(sys.argv)>=5):
    degree_rbf=int(sys.argv[4])
if(len(sys.argv)>=6):
    dropLabels=sys.argv[5:]

if(Dataset.find('xls')>=0):
    df=pd.read_excel(Dataset,dtype='unicode',skipinitialspace='true')
elif(Dataset.find('csv')>=0):
    df=pd.read_csv(Dataset,dtype='unicode',skipinitialspace='true')


if(dropLabels.__len__!=0):
        df=df.drop(dropLabels,axis=1)


sz,n_feat=df.shape
print("Dataset Read, Size=",sz)
print('Number of features=',n_feat-1)

df[LabelY]=LabelEncoder().fit_transform(df[LabelY])
X_train,X_test,y_train,y_test=train_test_split(df.drop(LabelY,axis=1),df[LabelY],test_size=floor(0.25*sz))

pkl.dump(X_train,open('X_train.pickle','wb'))
pkl.dump(y_train,open('y_train.pickle','wb'))
pkl.dump(X_test,open('X_test.pickle','wb'))
pkl.dump(y_test,open('y_test.pickle','wb'))

#print("\nDataset Split Saved...\n")

pca=PCA(6)
pca.fit(X_train)
pca_X_train=pca.transform(X_train)
pca_X_test=pca.transform(X_test)

print("PCA Dataset Ready...")
print('Number of features after PCA=',pca.n_components_)
models={'svm':SVC(kernel='linear'),'tree':DecisionTreeClassifier(),'rbf_svm':SVC(kernel='rbf',degree=degree_rbf,gamma='scale'),'logistic':LogisticRegression()}
clf=models[desired_model]
#print('Training...')
clf.fit(pca_X_train,y_train)

#print("Trained!\nCross Validating...\n")
cv = ShuffleSplit(n_splits=5, test_size=floor(0.25*sz), random_state=0)
scores=cross_val_score(clf,pca_X_train,y_train,cv=cv)
#print("\nPredicting...\n")
y_pred=clf.predict(pca_X_test)

print("\nValidation Accuracy=%.4f (+/- %0.4f)" % (scores.mean()*100,scores.std()*200))

print("Confusion Matrix:\n", confusion_matrix(y_test,y_pred,labels=[0,1]))
#print('\nClassification Report\n',classification_report(y_test,y_pred))
print("Testing Accuracy= ", accuracy_score(y_test,y_pred)*100)

pkl.dump(clf,open('classifier.pickle','wb'))
pkl.dump(pca_X_train,open('pca_X_train.pickle','wb'))
pkl.dump(pca_X_test,open('pca_X_test.pickle','wb'))
