import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

cancer=pd.read_csv('breast-cancer-wisconsin.csv')

print("------------Breast Cancer case study solved by KNN -----------")
print("Dimension Of Data Set :",cancer.shape)
print("The Decsription od Data :",cancer.describe)
print("Top 5 Rows of Data Set :\n",cancer.head)

cancer=cancer[cancer['BareNuclei']!='?']

x_train,x_test,y_train,y_test=train_test_split(cancer.loc[:,cancer.columns != 'CancerType'],cancer['CancerType'],stratify=
                                               cancer['CancerType'],random_state=66)


knn=KNeighborsClassifier(n_neighbors=31)

knn.fit(x_train,y_train)   
print("Accuracy Of Training  :{}".format(knn.score(x_train,y_train)))
# knn.predict(x_test,y_test)
print("Accuracy on Testing : {}".format(knn.score(x_test,y_test)))

