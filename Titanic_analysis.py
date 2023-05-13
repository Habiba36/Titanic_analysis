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
def main_function():
    available_functions = [
        '1- Describing the dataset attributes.',
        '2- Applying Data Cleaning.',
        '3- Data Visualization.',
        '4- Anomaly Detection (Finding Outliers).',
        '5-  Predictive Analytic (Decision Tree).',
        '6-  Predictive Analytic (KNN).',
    ]
    for function in available_functions:
        print(function)

    function_number = int(input())
    if 1 <= function_number <= 6:
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
    
    print('\n\n')
    functionNumber = main_function()


