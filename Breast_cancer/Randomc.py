import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#------------------------------FILE PATH------------
Input_path='breast-cancer-wisconsin.csv'
Output_path='Bcancer_made.csv'

#------------------------------Headers----------------------------
Headers=["CodeNumber","ClumpThickness","UniformityCellSize","UniformityCellShape",
         "MarginalAdhesion","SingleEpithelialCellSize","BareNuclei","BlandChromatin","NormalNucleoli",
         "Mitoses","CancerType"]

#--------------------Function which reads path in pandas data frame --------

def read_data(path):
    data=pd.read_csv(path)
    return data

#--------------------Function Which Returns Header ----------------

def get_headers(dataset):
    dataset.columns.values

#--------------------Function which add headers -----------------
    
def add_headers(dataset,headers):
    dataset.columns=headers
    return dataset

#----------------------Function which write data to csv----------

def data_file_to_csv():

    #Load the dataset into pandas data frame
    dataset=read_data()

    #add the headers to the loaded dataset
    dataset=add_headers(dataset,Headers)

    #Save the file
    dataset.to_csv(Output_path,index=False)
    print("File Saved")


#---------------------------Split The Dataset With Train Percentage--------------
    
def Split_data(dataset,train_percent,feature_headers,target_headers):
    train_x,test_x,train_y,test_y=train_test_split(dataset[feature_headers],dataset[target_headers],
                                                   train_size=train_percent)
    return train_x,test_x,train_y,test_y

#-------------------------Function removing the missing values----------------
def handle_missing_values(dataset,missing_values_header,missing_label):
    return dataset[dataset[missing_values_header]!= missing_label]


#------------------To train the random forest classifier feature and target --------- 
def Random_forest_classifier(features,target):
    clf=RandomForestClassifier()
    clf.fit(features,target)
    return clf

#----------------Function for displaying basic statistic of dataset---------------
def dataset_statistic(dataset):
    print(dataset.describe())

#------------------Main function from where execution starts----------------
    
def main():
    #load The data
    dataset=read_data(Input_path)

    #get basic statistics of loaded data set
    dataset_statistic(dataset)

    #Filter missing values
    dataset=handle_missing_values(dataset,Headers[6],'?')

    train_x,test_x,train_y,test_y=Split_data(dataset,0.7,Headers[1:-1],Headers[-1])

    #Train And Test Dataset Size Details
    print("Train_x Shape ::",train_x)
    print("Train_y Shape ::",train_y)
    print("Test_x Shape ::",test_x)
    print("Test_y Shape ::",test_y)

    #Creating random forest classifier instance
    trained_model=Random_forest_classifier(train_x,train_y)
    print("Trained Model ::",trained_model)
    predictions=trained_model.predict(test_x)

    for i in range(0,100):
        print("Actual Outcome ::{} and Predicted Outcome ::{}".format(list(test_y)[i],predictions[i]))

    
    print("Training Accuracy ::",accuracy_score(train_y,trained_model.predict(train_x)))
    print("Testing Accuracy ::",accuracy_score(test_y,predictions))
    print("Confusion Matrix ::",confusion_matrix(test_y,predictions))

#Starter
if __name__=="__main__":
    main()



    


    





