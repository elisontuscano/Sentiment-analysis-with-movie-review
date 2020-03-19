# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Train.csv')

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

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


