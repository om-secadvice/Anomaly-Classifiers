import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score,ShuffleSplit
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

LabelY='Label'
errmsg='USAGE: learn.py DesiredModel C'
models_list=['linear_svm','tree','logistic','forest']
argl=len(sys.argv)
if(argl>3):
    print(errmsg)
    exit(1)

desired_model='forest'
C=3.0


if(argl>=2 and sys.argv[1] in models_list):
    desired_model=sys.argv[1]

if(argl==3):
    C=float(sys.argv[3])
    



#Reading into pandas dataframe from excel file.
df_normal=pd.read_excel('df_normal.xlsx',skipinitialspace='true')
df_attack=pd.read_excel('df_attack.xlsx',skipinitialspace='true')

df_normal=df_normal.drop_duplicates()
df_attack=df_attack.drop_duplicates()

#Strip blankspaces from data
df_normal=df_normal.apply(lambda x: x.str.strip() if x.dtype == "object" else x).rename(columns=lambda x: x.strip())
df_attack=df_attack.apply(lambda x: x.str.strip() if x.dtype == "object" else x).rename(columns=lambda x: x.strip())

df=pd.concat([df_attack,df_normal])

sz,n_feat=df.shape
print("Dataset Read, Size=",sz)
print('Number of features=',n_feat-1)

df_temp=df
#Encoding Label Column to 0,1 for different classes. 
le=LabelEncoder()
df[LabelY]=le.fit_transform(df[LabelY])


X=df.drop(LabelY,axis=1)
y=df[LabelY]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=40,stratify=y)

df_train=pd.DataFrame(columns=df.columns,data=np.column_stack((X_train,y_train)))
df_test=pd.DataFrame(columns=df.columns,data=np.column_stack((X_test,y_test)))

df_train.to_excel('df_train_multiclass.xlsx',index=False)
df_test.to_excel('df_test_multiclass.xlsx',index=False)

#Centering data by mean i.e. subtracting mean from each data
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

   

models_list={'linear_svm':SVC(kernel='linear',C=C,class_weight='balanced'),
             'tree':DecisionTreeClassifier(class_weight='balanced'),
             'logistic':LogisticRegression(class_weight='balanced',C=C),
             'forest':RandomForestClassifier(class_weight='balanced'),   
             'BNB':BernoulliNB(),
             'Extree':ExtraTreeClassifier(),
             'Extrees':ExtraTreesClassifier(),
             'GNB':GaussianNB(),
             'KNN':KNeighborsClassifier(),
             'LDA':LinearDiscriminantAnalysis(),
             'LinearSVC':LinearSVC(multi_class='crammer_singer'),
             'logisticCV':LogisticRegressionCV(multi_class='multinomial'),
             'MLP':MLPClassifier(),
             'NearestCentroid':NearestCentroid(),
             'QDA':QuadraticDiscriminantAnalysis(),
             'RNC':RadiusNeighborsClassifier(),
             'GPC':GaussianProcessClassifier(multi_class = 'one_vs_rest')}




clf=models_list[desired_model]
print('\n\nTraining ',desired_model,' model...')

clf.fit(X_train,y_train)





#5 different splits of 25%,50% and taking average on validation score with some stdev
#This way we can effectively use all 75%  for training.(Leave One Out CV Method below)
cv = ShuffleSplit(n_splits=5, test_size=int(0.25*sz), random_state=40)
scores=cross_val_score(clf,X_train,y_train,cv=cv)
print('scores', scores)
print("Validation Accuracy=%.4f Std(+/- %0.4f)" % (scores.mean()*100,scores.std()*200))

y_pred=clf.predict(X_test)


conf_mat=confusion_matrix(y_test,y_pred)
print("\n\nConfusion Matrix:-\n",list(le.inverse_transform([0,1,2,3,4,5,6])),'\n',conf_mat)
print()
print("Testing Accuracy= ", clf.score(X_test,y_test)*100)

