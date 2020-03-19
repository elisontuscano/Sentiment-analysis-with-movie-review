# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle


# Importing the dataset
TrainDataset = pd.read_csv('Train.csv')
TrainDataset=shuffle(TrainDataset)
TestDataset = pd.read_csv('Test.csv')
TestDataset=shuffle(TestDataset)

#load the train and test split
X_train=TrainDataset['Review']
y_train=TrainDataset.iloc[:,1].values
X_test=TestDataset['Review']
y_test=TestDataset.iloc[:,1].values

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() 
corpus = []

review = re.sub('[^a-zA-Z]', ' ', X_train[0])
for i in range(0, 1000):
    #only keep alphabets remove rest
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    #turn all reviews into lowercase
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    #remove stopwords and do stemming
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #Lemmatize and merge the review together after making all the changes
    
    review = ' '.join([lemmatizer.lemmatize(word) for word in review])
    corpus.append(review)

corpus=dataset['Review']
# Creating the Bag of Words model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizor=TfidfVectorizer(stop_words='english', max_df=0.7 ,max_features=1500)
X = tfidf_vectorizor.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
y=y[:1000]

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
NaiveClassifier = GaussianNB()
NaiveClassifier.fit(X, y)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = NaiveClassifier, X = X, y = y, cv = 10)
accuracies.mean()

"""
testdataset = pd.read_csv('Test.csv')
X_test=testdataset['Review']
X_test=tfidf_vectorizor.transform(X_test).toarray()
y_test=testdataset.iloc[:,1].values

# Predicting the Test set results
y_pred = NaiveClassifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
"""

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 1500))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)



