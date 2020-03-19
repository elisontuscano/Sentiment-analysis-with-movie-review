import glob
import pandas as pd
import os

#Create positive and negative files in Train Folder
def createTrainPositive():
    df=[]
    #column=["Review","label"]
    for filename in glob.glob(os.path.join('train/pos/', '*.txt')):
        f = open(filename, 'r')
        content = f.read()
        df.append([content,'pos'])
    concatdf=pd.DataFrame(df)
    #concatdf.columns=column
    concatdf.to_csv('train/Positive.csv',index=None)

def createTrainNegative():
    df=[]
    #column=["Review","label"]
    for filename in glob.glob(os.path.join('train/neg/', '*.txt')):
        f = open(filename, 'r')
        content = f.read()
        df.append([content,'neg'])
    concatdf=pd.DataFrame(df)
    #concatdf.columns=column
    concatdf.to_csv('train/Negative.csv',index=None)

#Create Positive and negative files in Test Folders

def createTestPositive():
    df=[]
    #column=["Review","label"]
    for filename in glob.glob(os.path.join('test/pos/', '*.txt')):
        f = open(filename, 'r')
        content = f.read()
        df.append([content,'pos'])
    concatdf=pd.DataFrame(df)
    #concatdf.columns=column
    concatdf.to_csv('test/Positive.csv',index=None)

def createTestNegative():
    df=[]
    #column=["Review","label"]
    for filename in glob.glob(os.path.join('test/neg/', '*.txt')):
        f = open(filename, 'r')
        content = f.read()
        df.append([content,'neg'])
    concatdf=pd.DataFrame(df)
    #concatdf.columns=column
    concatdf.to_csv('test/Negative.csv',index=None)
    

#Combine the files and create Train and Test CSV
def createTrain():
    df=[]
    column=["Review","label"]
    for filename in glob.glob(os.path.join('train/', '*.csv')):
        content=pd.read_csv(filename,header=None)
        df.append(content)
    concatdf=pd.concat(df,axis=0)
    cancatdf=concatdf.sample(frac=1)
    concatdf.columns=column
    concatdf.to_csv('Train.csv',index=None)

def createTest():
    df=[]
    column=["Review","label"]
    for filename in glob.glob(os.path.join('test/', '*.csv')):
        content=pd.read_csv(filename,header=None)
        df.append(content)
    concatdf=pd.concat(df,axis=0)
    cancatdf=concatdf.sample(frac=1)
    concatdf.columns=column
    concatdf.to_csv('Test.csv',index=None)

createTrainPositive()
createTrainNegative()
createTestPositive()
createTestNegative()
createTrain()
createTest()


    
