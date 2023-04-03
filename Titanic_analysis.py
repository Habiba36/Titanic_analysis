# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 17:40:55 2022

@author: DELL
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_blobs
from sklearn.tree import export_graphviz, DecisionTreeRegressor
from sklearn import tree, neighbors
from numpy import random, where, ndarray
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.neighbors import LocalOutlierFactor
from numpy import where, quantile, random, ndarray
#from textblob.classifiers import NaiveBayesClassifier
def describing_dateset_attributes():
    print('''
Titanic Dataset Attributes Description:
 1. PassengerId: Unique Id of a passenger
 2. Survived: If the passenger survived (0 = No, 1 = Yes)
 3. PClass: Passenger Class (1 = 1st class, 2 = 2nd class, 3 = 3rd class)
 4. Name: Name of the passenger
 5. Sex: Male/Female
 6. Age: Passenger age in years
 7. SibSp: No of siblings/spouses aboard
 8. Parch: No of parents/children aboard
 9. Ticket: Ticket number
 10. Fare: Passenger fare
 11. Cabin: Cabin number
 12. Embarked: Port of Embarkation (C = Cherboung; Q = Queenstown; S = Southampton)
''')

    data = pd.read_csv('titanic.csv')
    print(data.describe())

def data_cleaning():
    # Firstly data cleaning:
    # As we can see that age and cabin have missing values and we can adjust the age values
    # by guessing the age as enough data is available to make predictions but in the cabin
    # major data is missing so we will drop this
    # Embarked col has only 2 null cells so we can drop these 2 rows
    data = pd.read_csv('titanic.csv')
    data.info()
    data.drop('Cabin', axis=1, inplace=True)
    data.dropna(axis=0, subset=['Embarked'], inplace=True)  # axis = 0;deletes row, =1 deletes col
    data['Age'].fillna(math.trunc(data['Age'].mean()), inplace=True)
    data.to_csv('titanic_modified.csv', index=False)


def data_visualization():
    # Data visualization

    databefore = pd.read_csv('titanic.csv')
    sns.boxplot(x='Pclass', y='Age', data=databefore)
    # boxplot shows the average of passengers' ages and the age pattern with respect to a Pclass
    # it can be used in guessing the null values in the Age column
    plt.show()
    ##############
    data = pd.read_csv("titanic_modified.csv")
    surCount = [(data['Survived'] == 0).sum(), (data['Survived'] == 1).sum()]
    plt.pie(surCount, labels=['Dead', 'Survived'], autopct="%1.0f%%")
    plt.show()
    # piechart shows the percentages of survival vs deaths

    ###############
    w = 0.2
    x = ["Female", "Male"]
    survived = [((data['Survived'] == 1) & (data['Sex'] == 'female')).sum(),
                ((data['Survived'] == 1) & (data['Sex'] == 'male')).sum()]

    dead = [((data['Survived'] == 0) & (data['Sex'] == 'female')).sum(),
            ((data['Survived'] == 0) & (data['Sex'] == 'male')).sum()]

    index = np.arange(len(x))  # [0,1]

    plt.bar(index + 0.1, dead, w, label="Dead")
    plt.bar(index - 0.1, survived, w, label="Survived")

    plt.ylabel("Number of passengers")
    plt.xticks(index, x)
    plt.legend()
    plt.show()

    # Clustered bar chart is a comparison chart, it shows the survival distribution on
    # male and female passengers.
    # The ratio of survival is smaller for the male passengers.
    # for the female passengers the ratio of survival is bigger than death,it is more than twice.
    # for the male passengers the ratio of death is much bigger than survival,
    # it is more than 4 times.
    ###############

    plt.hist(data.Age, bins=[0, 10, 20, 30, 40, 50, 60, 70, 80], color='blue', edgecolor='black')
    plt.xlabel('Age of passengers')
    plt.ylabel('No of passengers')
    plt.title('Age Distribution')
    # histogram shows the frequency distribution; it divides the ages into bins
    # and shows a bar plot of the number of passengers in each bin

# most of the passengers of 1st class their ages ranges from(28 to 50)
# while those of the 2nd class their ages ranges from (28 to 38)
# while those of the third class most of them their ages ranges from (20 to 25)
# the observation on the pie chart is that the number of people who died was more than the number of people who survived from sinking in the ocean.
# the observation on the histogram is that most of the passengers were on their middle age ranges between(20 to 40)

def anomaly_detection():
    data = pd.read_csv("titanic_modified.csv")
    sub_data = data.head(70)
    random.seed(10)
    x = sub_data[['Fare', 'Age']].values
    plt.scatter(x[:, 0], x[:, 1])
    plt.show()
    lof = LocalOutlierFactor(n_neighbors=30, contamination=0.1)
    y_pred = lof.fit_predict(x)
    lofs_index = where(y_pred == -1)
    values = x[lofs_index]
    model = LocalOutlierFactor(n_neighbors=30)
    model.fit_predict(x)
    lof = model.negative_outlier_factor_
    thresh = quantile(lof, 0.1)
    print(thresh)
    index = where(lof <= thresh)
    values = x[index]
    plt.scatter(x[:, 0], x[:, 1])
    plt.scatter(values[:, 0], values[:, 1], color='r')
    plt.show()
    #data = pd.read_csv('titanic.csv')
    # data.info()
    # data.drop('Cabin', axis=1, inplace=True)
    # data.dropna(axis=0, subset=['Embarked'], inplace=True)  # axis = 0;deletes row, =1 deletes col
    # data['Age'].fillna(math.trunc(data['Age'].mean()), inplace=True)
    # data.to_csv('titanic_modified.csv', index=False)
    # data.info()
   #  data = pd.read_csv("titanic_modified.csv")
   #  broken_dataset = data.copy()
   #  #random.seed(7)
   # # data, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.3, center_box=(20, 5))
   #  plt.scatter(broken_dataset[:,0], broken_dataset[:,1])
   # # calculate the distance from each point to its closest neighbour
   #  neigh = NearestNeighbors(n_neighbors=2)
   #  nbrs = neigh.fit(broken_dataset)
   #  distances, indices = nbrs.kneighbors(broken_dataset)
   #  distances = np.sort(distances, axis=0)
   #  distances = distances[:, 1]
   #  # plt.plot(distances)
   #  dbscan = DBSCAN(eps=0.18, min_samples=10)
   #  print(dbscan)
   #  pred = dbscan.fit_predict(broken_dataset)
   #  anom_index = where(pred == -1)
   #  values = broken_dataset[anom_index]
   #  plt.scatter(broken_dataset[:, 0], broken_dataset[:, 1])
   #  plt.scatter(values[:, 0], values[:, 1], color='r')
   #  plt.show()


def predictive_analytics_decision_tree():
    data = pd.read_csv('titanic_modified.csv')
    data['Sex'] = data.Sex.replace({'male': 0, 'female': 1})
    data['Embarked'] = data.Embarked.replace({'S': 1, 'C': 2, 'Q': 3})
    x: ndarray = data[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values
    y = data['Survived']
    model: DecisionTreeRegressor = tree.DecisionTreeRegressor()
    model.fit(x, y)
    predictions = model.predict(x)
    print(model.feature_importances_)

    for index in range(len(predictions)):
        print('Actual : ', y[index], 'Predicted:	', predictions[index])

    plt.figure(figsize=(300, 100))
    tree.plot_tree(model, fontsize=10, feature_names=x)
    export_graphviz(model, out_file='DecisionTree.dot')


def predictive_analytics_knn():
    data = pd.read_csv('titanic_modified.csv')
    data['Sex'] = data.Sex.replace({'male': 0, 'female': 1})
    data['Embarked'] = data.Embarked.replace({'S': 1, 'C': 2, 'Q': 3})

    x = data[['PassengerId', 'Survived', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values
    y = data['Pclass']

    model = neighbors.KNeighborsRegressor(n_neighbors=5)
    model.fit(x, y)
    predictions = model.predict(x)

    for index in range(len(predictions)):
        print('Actual: ', y[index], '  ', 'Predicted :', predictions[index], '  ', 'Weight', x[index, 0])


def printing_sentences_in_lowercase():
    # Converting the words of each sentence to a lowercase words and printing it
    # Getting the Sentences from the excel file
    data = pd.read_csv('TextDataset.csv')
    # Adding all the sentences to a list
    sentences = list(data['text'])
    for sentence in sentences:
        sentence = sentence.lower()
        print(sentence)


def printing_sentences_in_uppercase():
    # Converting the words of each sentence to uppercase words and printing it
    # Getting the Sentences from the excel file
    data = pd.read_csv('TextDataset.csv')
    # Adding all the sentences to a list
    sentences = list(data['text'])
    for sentence in sentences:
        sentence = sentence.upper()
        print(sentence)


def removing_punctuation_in_each_sentence():
    # Getting the Sentences from the excel file
    data = pd.read_csv('TextDataset.csv')
    # Adding all the sentences to a list
    sentences = list(data['text'])
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for sentence in sentences:
        print("\nBefore removing punctuation: "+sentence)
        for word in sentence:
            if word in punctuations:
                sentence = sentence.replace(word, "")
        print("After removing punctuation: " + sentence)


# Tokenizing Sentences (used in different functions)
def tokenizing_sentences():
    # Getting the Sentences from the excel file
    data = pd.read_csv('TextDataset.csv')
    # Adding all the sentences to a list
    sentences = list(data['text'])
    tokenized_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        sentence_words = word_tokenize(sentence)
        tokenized_sentences.append(sentence_words)
    return tokenized_sentences


def print_tokenized_sentences():
    tokenized_sentences = tokenizing_sentences()
    for tokenized_sentence in tokenized_sentences:
        print(tokenized_sentence)


def removing_stopwords():
    # gets the stop words found in english in a set
    stops = set(stopwords.words('english'))
    tokenized_sentences = tokenizing_sentences()
    # Removes the stop words found in a sentence and printing the words in a list without the stop words
    for sentence in tokenized_sentences:
        print("\nBefore removing stop words: ", sentence)
        for word in sentence:
            if word in stops:
                sentence.remove(word)
        print("After removing stop words: ", sentence)


def stemming_words_in_each_sentence():
    stemmer = PorterStemmer()
    tokenized_sentences = tokenizing_sentences()

    for sentenceWords in tokenized_sentences:
        stemmed_sentence = []
        print("\nwords before stemming: ", sentenceWords)
        for word in sentenceWords:
            stemmed_sentence.append(stemmer.stem(word))
        print("words after stemming: ", stemmed_sentence)


def lemmatizing_words_in_each_sentence():
    lemmatizer = WordNetLemmatizer()
    tokenized_sentences = tokenizing_sentences()

    for sentence in tokenized_sentences:
        lemmatized_sentence = []
        print("\nwords before lemmatization: ", sentence)
        for word in sentence:
            lemmatized_sentence.append(lemmatizer.lemmatize(word))
        print("words after lemmatization: ", lemmatized_sentence)


def sentiment_analysis():
    # Getting the Sentences from the excel file
    data = pd.read_csv('TextDataset.csv')
    # Adding all the sentences to a list
    sentences = list(data['text'])
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    for sentence in sentences:
        print('\n'+sentence)
        print(sia.polarity_scores(sentence))


def classification():
    # Getting the Sentences from the excel file
    data = pd.read_csv('TextDataset.csv')
    # Adding all the sentences to a list
    sentences = list(data['text'])
    train = [("what an amazing weather", 'pos'),
             ("this is an amazing idea!", 'pos'),
             ("I feel very good about these ideas.", 'pos'),
             ("what an awesome view", 'pos'),
             ("I really like the zoom app cause it allows us to be there for each other in a time like now when many things Are uncertain.", 'pos'),
             ("this is my best performance.", 'pos'),
             ('Great sacrament meeting. I am very grateful for this opportunity to listen from home occasionally.', 'pos'),
             ('I like it because you can have a host and share your screen and you can do cool thing.', 'pos'),
             ('Well, we walk and fly on the shoulders of Giants!', 'pos'),
             ('I do not like this place', 'neg'),
             ('I am tired of this stuff.', 'neg'),
             ('The speaker was hard to understand at times because of glitches.', 'neg'),
             ('I can\'t deal with all this tension', 'neg'),
             ('he is my sworn enemy!', 'neg'),
             ('my friends are horrible', 'neg'),
             ]
    test = [
        ("i cannot download the app.customer support can’t even help.", "neg"),
        ("could you please help to fix it.thank you for your help.", "pos"),
        ("trying to join zoom is more difficult than it should be.", "neg"),
        ("this is a great resource conference", "pos"),
        ("very clear reception with great sound!", "pos"),
    ]
  #  c1 = NaiveBayesClassifier(train)
    for sentence in sentences:
        print("\n"+sentence)
  #      print("Classified: "+c1.classify(sentence))
   # print("\nClassification Accuracy: ", c1.accuracy(test))


def main_function():
    available_functions = [
        '1- Describing the dataset attributes.',
        '2- Applying Data Cleaning.',
        '3- Data Visualization.',
        '4- Anomaly Detection (Finding Outliers).',
        '5-  Predictive Analytic (Decision Tree).',
        '6-  Predictive Analytic (KNN).',
        '7- Printing words in each sentence in lowercase.',
        '8- Change words in each sentence in uppercase.',
        '9- Remove punctuations in each sentence.',
        '10- Tokenizing each Sentence.',
        '11- Removing stop words.',
        '12- Stem words in each sentence.',
        '13- Lemmatize words in each sentence.',
        '14- Sentiment analysis',
        '15- Classification',
        '16- Exit Program.',
    ]
    for function in available_functions:
        print(function)

    function_number = int(input())
    if 1 <= function_number <= 16:
        return function_number
    else:
        print("*****Please enter a valid number*****")
        return main_function()


functionNumber = main_function()

while not functionNumber == 16:
    if functionNumber == 1:
        describing_dateset_attributes()
    elif functionNumber == 2:
        data_cleaning()
    elif functionNumber == 3:
        data_visualization()
    elif functionNumber == 4:
        anomaly_detection()
    elif functionNumber == 5:
        predictive_analytics_decision_tree()
    elif functionNumber == 6:
        predictive_analytics_knn()
    if functionNumber == 7:
        printing_sentences_in_lowercase()
    elif functionNumber == 8:
        printing_sentences_in_uppercase()
    elif functionNumber == 9:
        removing_punctuation_in_each_sentence()
    elif functionNumber == 10:
        print_tokenized_sentences()
    elif functionNumber == 11:
        removing_stopwords()
    elif functionNumber == 12:
        stemming_words_in_each_sentence()
    elif functionNumber == 13:
        lemmatizing_words_in_each_sentence()
    elif functionNumber == 14:
        sentiment_analysis()
    elif functionNumber == 15:
        classification()
    print('\n\n')
    functionNumber = main_function()


