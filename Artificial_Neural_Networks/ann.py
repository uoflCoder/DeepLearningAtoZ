# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
#Import the dataset Churn_Modeling.csv file
dataset = pd.read_csv('Churn_Modelling.csv')

#X values, get all the rows and get relevant columns (3-12)
X = dataset.iloc[:, 3:13].values

#y value, get all the rows and get the relevant output column (13)
y = dataset.iloc[:, 13].values

# Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
#Label encode the countries to be distinct integer values
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
#Label encode male and female (1,0) (0,1)
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#One hot encoder, transform the column Geography (1,0,0) (0,1,0) (0,0,1) to three columns
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

#Get rid of the first column keep all rows because based on columns 2 and 3 you could deduce
#column 1 without having column 1
X = X[:, 1:]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
#NOTE THIS CODE, this is easy reusable code that gives in this example 80% to the training set and 20% to the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
#Easy feature scaling reusable code to scale the columns in the X training and X test set
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!
"""
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer

#Add the first hidden layer to have 6 nodes, let the initial values be close to 0 but 'uniform' 
#use the rectifier function ('relu') and note that the number of input features is 11
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
#Make a second hidden layer and since this is the second layer you do not need
#to state the number of input dimension but everything else remains the same as above
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
#Create an output layer that has one output and give it an initial value close to 0
#and get the activation function to be sigmoid because it is between 0 and 1 probability float value
#and we want the probability that someone will leave the bank 
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
#This is the one I will have to research most
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set, 
#use input X_train, output y_train, batch size (10 at a time and update weights per 10), run throught the whole set 100 times
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
#Predict the values on the test results
y_pred = classifier.predict(X_test)
#Return a true or false value instead of probabilities
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
#Get a 2X2 (0,0) (0,1) (1,0) (1,1) matrix that shows how many predictions you got right
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#accuracy of mine
(1544 + 51)/2000

"""
"""
Geography: France
Credit Score 600
Gender Male
Age 40
Tenure 3
Balance 60000
Number of Products 2
Has Credit Card Yes
Is Active Member Yes
Estimated Salary: 50000

new_prediction = classifier.predict(sc.transform(np.array([[0.000, 0.000, 600.000, 1.000, 40.000, 3.000, 60000.000, 2.000, 1.000, 1.000, 50000.000]
])))
"""

#Part 4, evaluating, improving, and tuning the ANN

#Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

#Build the standard classifier to be used later in the code
def build_classifier():
    from keras.models import Sequential
    from keras.layers import Dense

    # Initialising the ANN
    classifier = Sequential()
    
    # Adding the input layer and the first hidden layer
    
    #Add the first hidden layer to have 6 nodes, let the initial values be close to 0 but 'uniform' 
    #use the rectifier function ('relu') and note that the number of input features is 11
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    
    # Adding the second hidden layer
    #Make a second hidden layer and since this is the second layer you do not need
    #to state the number of input dimension but everything else remains the same as above
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    
    # Adding the output layer
    #Create an output layer that has one output and give it an initial value close to 0
    #and get the activation function to be sigmoid because it is between 0 and 1 probability float value
    #and we want the probability that someone will leave the bank 
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    
    # Compiling the ANN
    #This is the one I will have to research most
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return classifier

#Build the KerasClassifier with a batch size of 10 and a epoch of 100 times
classifier = KerasClassifier(build_fn = build_classifier(), batch_size = 10, nb_epoch= 100)
#Set the accuracies to be a cross_validation using the classifier built and the X_train and y_train with a 10 fold
#cross validation and use all cpus
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1 )