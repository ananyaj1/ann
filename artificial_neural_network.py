# Artificial Neural Network

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf

tf.__version__ # print the tensorflow version that we are using

################################ SET UP DATASET / TRAINING ################################
# Import dataset
dataset = pd.read_csv('Churn_Modelling.csv')
# all data (only relevant categories, exclude customer identifiers and expected value)
X = dataset.iloc[:, 3:-1].values # only grab relevant inputs for prediction
# expected value (whether customer left bank or not)
y = dataset.iloc[:, -1].values # grab all sample data

#TESTING
#print(X)
#print(y)

# transform categorical data like gender into 0 for female, 1 for male

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2]) #gender is column 2, grab all rows

# TESTING GENDER
#print(X)

# One Hot Encoding the "Geography" column
# One Hot Encoding: will split the geography column into multiple columns
# binary values for which country the data is 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# Geography is in column 1
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# TESTING GEOGRAPHY 
#print(X)
# France: 1 0 0, Spain: 0 0 1, Germany: 0 1 0

# Split the dataset into training data and testing data 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
# Necessary for a better training model 
# Algorithm only sees numbers: avoid the underlying assumption that higher number > lower number
# Apply feature scaling to ALL categories, even yes/no categories
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

################################################################################################
################################ BUILDING THE ANN ##############################################

# Create a sequence of layers for the network
ann = tf.keras.models.Sequential()
# Sequential class has a method for adding layers: 
# Specifically use the Dense layer
# Dense layer: every 'neuron' recieves inputs from all neurons of previous layer

# FIRST HIDDEN LAYER
ann.add(tf.keras.layers.Dense(units=5, activation='relu')) 
# 5 hidden neurons(arbitrarily chosen), Rectifier activation function

# SECOND HIDDEN LAYER (fully connected to first hidden layer)
ann.add(tf.keras.layers.Dense(units=6, activation='relu')) 

# OUTPUT LAYER (fully connected to second hidden layer)
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) 
# since our output is a binary value (0 or 1), we only need 1 dimension
# Sigmoid activiation function will give us the probability of the outcome being 1

ann.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])
"""
adam optimizer will perform stocastic gradient descent:
reduce the loss between predicted value and expected value
loss function: since we are predicting a binary value,
we will use binary_crossentropy. if we were predicting a non-binary value,
we would use categorical cross entropy. 
"""

################################################################################################
################################ TRAINING THE ANN ##############################################

# batch size is the batch of predictions compared to expected value
# epochs chosen arbitrarily
ann.fit(X_train, y_train, batch_size=32, epochs=100)

# accuracy is hovering around 86%, pretty good 

################################################################################################
############################ PREDICTING USING THE ANN ##########################################
"""
Can we use our trained model to predict if a certain customer will leave the bank or not? 

Solution: Scale the dataset and predict if this customer will leave or not:
"""
# will return the probability of the customer leaving the bank
ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))

# TESTING PREDICTION
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
# [[0.02162948]] => low probability of leaving bank

# predict using the testing dataset
y_pred = ann.predict(X_test) 
# return a true or false value
y_pred = (y_pred > 0.5)
# create & print an array of the predicted value next to the expected value
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
################################################################################################
