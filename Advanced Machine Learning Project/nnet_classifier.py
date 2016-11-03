from __future__ import print_function
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.utils import np_utils
from keras.layers import recurrent
from math import sqrt
import pylab as plt
import seaborn as sns
import matplotlib.pyplot as plt

class nnet_classifer():
    def __init__(self):        
        # Define Hyperparameters
        np.random.seed(1337)  # for reproducibility
        # Initialize the models
        self.lstm_model = Sequential()
        self.bs_model = Sequential()
        # Width of the matrix: i.e. number of features
        self.inputLayerSize = 0
        # Number of unique classes in the output
        self.outputLayerSize = 0
        # Width of the hidden layer
        self.hiddenLayerSize = 0
        # Number of batches to take in for processing
        self.batch_size = 10
        # Number of epochs to run
        self.nb_epoch = 3
        # Window size of the padding
        self.window_size = 10
        # Initialize train/test data
        self.X_train = 0
        self.X_test = 0
        self.y_train = 0
        self.y_test = 0
        # Initialize y_test and y_train binary arrays
        self.y_binary_train = 0
        self.y_binary_test = 0
        # Initialize preprocess data
        self.preprocessed_X_Train = np.zeros((0, 0, 0))
        self.preprocessed_X_Test = np.zeros((0, 0, 0))
        # Set the transform to binary flag
        self.transform_to_binary_flag = False
        # Set history
        self.history = None
        # Set architecture
        self.architecture = None
        # Set score
        self.score = 0
        self.loss = 0
        self.val_loss = 0
        self.acc = 0
        self.val_acc = 0
        self.window_size_flag = False
        self.sma_loss = 0
        self.sma_acc = 0
        
    def transform_to_binary(self, X_test, y_train, y_test):
        " Transforms the output data into a binary array"
        '''
        Input: [1,2,3]
        Output: [[1, 0, 0],
                 [0, 1, 0],
                 [0, 0, 1]]
        Note: It does not return anything, it updates the initialized parameters
        '''
        self.y_test = y_test
        self.y_train = y_train
        self.X_test = X_test
        print('Inititialized X_test!')
        self.outputLayerSize = len(np.unique(y_train))
        # Using the inbuilt to_categorical function to do it
        self.y_binary_test = to_categorical(self.y_test)
        self.y_binary_train = to_categorical(self.y_train)
        # The output contains an added row so we have 
        # to avoid that by keeping all other columns except first
        # The output layer size explains the number of expected outputs
        self.y_binary_test = self.y_binary_test[:,1:self.outputLayerSize+1]
        self.y_binary_train = self.y_binary_train[:,1:self.outputLayerSize+1]
        # Set the flag to true once the function runs
        self.transform_to_binary_flag = True
        print('Completed: y_train and y_test transformed to binary!')

    def preprocess_input(self, window_size, x, y):
        "  "
        '''
        Input: Takes in the input of window_size, x's and y's for training or test data
        Output: Outputs a tensor with dimensions (Number_of_samples, window_size, inputLayerSize)
        '''
        # Inititialize local variables sample size and feature size as the dimensions of input data
        sample_size = x.shape[0]
        feature_size = x.shape[1]
        # Initialize the tensor of zeros which we will fill in later
        preprocessed_X = np.zeros((sample_size, window_size, feature_size))
        # Output the dimensions of the expected dimensions
        print("Number of samples: ",len(y), "\n")
        # Loop over indices of y
        for y_index in range(len(y)):
            # Loop over indexes of x and shift the window every iteration and build our tensor
            for each_x_index, x_index in zip(range(window_size), range(y_index-window_size+1, y_index+1)):
                # Ignore the negative indices of x and update the tensor values only if, 
                # the index value is greater than 0 
                if x_index >=0:
                    # Maintain y_index and x_index and update the index where x > 0 with,
                    # Value of x at the given index
                    preprocessed_X[y_index, each_x_index,:] =  x[x_index]
        # Return the preprocessed value once the computation is over
        return preprocessed_X

    def initialize_input_parameters(self, X_train, y_train, hidden_layer_size, batch_size):
        ''' Inititializes parameters based on the given inputs'''
        # Width of the data frame is the size of the input layer
        self.inputLayerSize = X_train.shape[1]
        # Get the unique number of output classes
        self.outputLayerSize = len(np.unique(y_train))
        # Set the hidden layer size
        self.hiddenLayerSize = hidden_layer_size
        # Set the batchsize
        self.batch_size = batch_size
        print('Input Parameters initialized!')
        
    def fit(self, X_train, y_train, architecture = 'mlp', 
        window_size = 10, hidden_layer_size = 25, update_method = 'adam',
        activation_function = 'relu', dropout = 0.5, nb_epoch = 3, batch_size=32,
        validation_split=0.1, batch_normalization=False, verbose=0):
        '''
        Prerequisites: Please transform_to_binary before calling this function
        This function will build the model.

        Parameters:
        architecture = Type of architecture 'mlp', 'lstm', default='mlp'
        hidden_layer_size = Width of hidden layer, default=25
        update_method = 'adadelta','adam','adagrad', default= 'adam'
        window_size = Width of the window, default=10
        activation_function = Type of activation functions, default='relu'
        dropout = dropout values after a given threshold, default = 0.5
        Other parameters defaults: dropout = 0.5, nb_epoch = 3, batch_size=32, validation_split=0.1
        batch_normalization: Add a batch normalization layer, default = False
        verbose: print processing, default: 0
        '''
        # Check if transform to binary flag is called.
        # If it is allow building the model else throw an error.
        self.architecture = architecture
        self.nb_epoch = nb_epoch

        if self.transform_to_binary_flag == True:
            if self.window_size_flag == False:
            # Call the initialize input parameters method to update parameters
                self.initialize_input_parameters(X_train, y_train, hidden_layer_size, batch_size)
                self.window_size_flag = True
            
            # Check if architecture is LSTM:
            if architecture == 'lstm':
                # Call preprocess method
                print('Generating preprocessed_X_Train ... \n')
                self.preprocessed_X_Train = self.preprocess_input(window_size, X_train, y_train)
                print('Generating preprocessed_X_Test ... \n')
                self.preprocessed_X_Test = self.preprocess_input(window_size,self.X_test,self.y_test)
                print('Building LSTM Model ... \n')
                self.lstm_model.add(recurrent.LSTM(window_size, 
                                    input_shape=(window_size, self.inputLayerSize)))
                # Check if batch normalization is enabled
                if batch_normalization == True:
                    # Add batch normalization layer with default values
                    self.lstm_model.add(BatchNormalization(epsilon=1e-06,
                                                        mode=0,
                                                        axis=-1,
                                                        momentum=0.9,
                                                        weights=None,
                                                        beta_init='zero',
                                                        gamma_init='one'))
                self.lstm_model.add(Dense(self.hiddenLayerSize))
                self.lstm_model.add(Activation(activation_function))
                self.lstm_model.add(Dropout(dropout))
                self.lstm_model.add(Dense(self.outputLayerSize, activation='softmax'))

                self.lstm_model.compile(loss='categorical_crossentropy',
                                        optimizer=update_method,
                                        metrics=['accuracy'])
                # Call the LossHistory class to collect loss and accuracy for every batch
                # process in the epoch
                self.history = LossHistory()

                self.lstm_model.fit(self.preprocessed_X_Train, self.y_binary_train,
                                    nb_epoch=nb_epoch, batch_size=batch_size,
                                    verbose=verbose, validation_split=validation_split, 
                                    callbacks=[self.history])
                print('LSTM Model built !')
                self.window_size_flag = False

            # Check if architecture is multi layer perceptron
            elif architecture == 'mlp':
                print('Building mlp model ...')
                # Add the first layer which has the dimensions (inputLayerSize,hiddenLayerSize)
                self.bs_model.add(Dense(hidden_layer_size, input_shape=(self.inputLayerSize,)))
                # Check if batch normalization is enabled
                if batch_normalization == True:
                    # Add batch normalization layer with default values
                    self.bs_model.add(BatchNormalization(epsilon=1e-06,
                                                        mode=0,
                                                        axis=-1,
                                                        momentum=0.9,
                                                        weights=None,
                                                        beta_init='zero',
                                                        gamma_init='one'))
                # Define the activation function
                self.bs_model.add(Activation(activation_function))
                # Define the dropout
                self.bs_model.add(Dropout(dropout))
                # Define the next layer based on the number of classes
                self.bs_model.add(Dense(self.outputLayerSize))
                # Output the result into softmax function to get probabilty array
                self.bs_model.add(Activation('softmax'))
                # Build the model
                self.bs_model.compile(loss='categorical_crossentropy',
                            optimizer=update_method,
                            metrics=['accuracy'])
                # Build the history from the data
                self.history = LossHistory()

                self.bs_model.fit(X_train, self.y_binary_train,
                    nb_epoch=nb_epoch, batch_size=batch_size,
                    verbose=verbose, validation_split=validation_split,
                    callbacks=[self.history])
                print('MLP Model built !')
                self.window_size_flag = False
            # If unknown architecture is inserted throw an error
            else:
                print("Unknown architecture inserted! Please check your input")

        else:
            print("Please transform y_train and y_test to binary using the transform_to_binary function!")
        
    def predict(self, X_test):
        # Compute the scores based on the type of model
        if self.architecture == 'lstm':
            self.score = self.lstm_model.evaluate(self.preprocessed_X_Test, self.y_binary_test,
                       batch_size=self.batch_size, verbose=1)
            print('Test loss LSTM:', self.score[0])
            print('Test accuracy LSTM:', self.score[1])
        # Check if architecture is multi layered perceptron
        elif self.architecture == 'mlp':
            self.score = self.bs_model.evaluate(X_test, self.y_binary_test,
                       batch_size=self.batch_size, verbose=1)
            print('Test loss mlp:', self.score[0])
            print('Test accuracy mlp:', self.score[1])
        else:
            print('Check fit method!')
            
        return self.score

    def plot_metric_epoch(self, metric_type='accuracy', figsize=(7,4)):
        '''
        function that plots the given metric with a default figure size of (7,4)
        metric_type: can take values: accuracy, loss, val_acc, val_loss
        '''
        sns.set_style('white')
        plt.figure(figsize=figsize)
        if metric_type == 'accuracy':
            plt.plot(self.history.history.values()[0])
        elif metric_type == 'loss':
            plt.plot(self.history.history.values()[1])
        elif metric_type == 'val_acc':
            plt.plot(self.history.history.values()[2])
        elif metric_type == 'val_loss':
            plt.plot(self.history.history.values()[3])
        else:
            print(" Wrong metric_type! Check your input .. ")
        plt.xlabel('epochs')
        plt.ylabel(metric_type)
        plt.title(metric_type+str(' plot'))
        plt.show()

    def movingaverage (self, values, window):
        weights = np.repeat(1.0, window)/window
        sma = np.convolve(values, weights, 'valid')
        return sma

    def plot_metric_batch(self, metric_type='accuracy', sma=500, figsize=(7,4)):
        '''
        function that plots the given metric with a default figure size of (7,4)
        metric_type: can take values: accuracy, loss
        sma explains the movingaverage: default 500
        '''
        # Get parameters from history
        self.loss = self.history.loss
        self.acc = self.history.acc
        
        simple_ma = sma
        self.sma_loss = self.movingaverage(self.loss, simple_ma)
        self.sma_acc = self.movingaverage(self.acc, simple_ma)
        x_label = str(self.nb_epoch) + ' epochs at ' + str(np.round(len(self.sma_acc)/self.nb_epoch)) + ' iterations per epoch'
        
        sns.set_style('white')
        plt.figure(figsize=figsize)
        if metric_type == 'loss':
            plt.plot(self.sma_loss,color='red', label='Train Loss')
        elif metric_type == 'accuracy':
            plt.plot(self.sma_acc, label="Train Accuracy")
        else:
            print(" Wrong metric_type! Check your input .. ")
        plt.xlabel(x_label)
        plt.ylabel(metric_type)
        plt.title(metric_type+str(' plot'))
        plt.show()

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []

    def on_batch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_loss.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
