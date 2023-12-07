#!/usr/bin/env python
# coding: utf-8

# In[1]:


#*******************************************************************************************
 #
 #  File Name:  AlphabetSoupCharityOptimizationSearchFunctions.py
 #
 #  File Description:
 #      This Python script, AlphabetSoupCharityOptimizationSearchFunctions.py, contains 
 #      generic Python functions for completing common neural network configuration
 #      tasks.  Here is the list:
 #
 #      ReturnColumnSeriesAndSortedValueCountList
 #      SetFeaturesInteger
 #      ReturnFeaturesInteger
 #      ReturnBinnedDataFrameFromOneColumn
 #      ReturnBinnedDataFrame
 #      ReturnNeuralNetworkXYParameters
 #      ReturnNeuralNetworkModel
 #      ReturnBestModelDictionary
 #
 #
 #  Date            Description                             Programmer
 #  ----------      ------------------------------------    ------------------
 #  12/02/2023      Initial Development                     N. James George
 #
 #******************************************************************************************/

import PyConstants as constant
import PyFunctions as function
import PyLogConstants as log_constant
import PyLogFunctions as log_function
import PyLogSubRoutines as log_subroutine
import PySubRoutines as subroutine

import AlphabetSoupCharityOptimizationSearchConstants as local_constant

from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import keras_tuner as kt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd


# In[2]:


CONSTANT_LOCAL_FILE_NAME \
    = 'AlphabetSoupCharityOptimizationSearchFunctions.py'


# In[3]:


def ReturnColumnSeriesAndSortedValueCountList \
        (inputDataFrame,
         columnNameString):
    
    try:

        tempSeries \
            = inputDataFrame[columnNameString] \
                .value_counts() \
                .sort_values \
                    (ascending = False)
        
        tempSeries \
            .name = columnNameString

        log_function \
            .DebugReturnObjectWriteObject \
                (tempSeries)
        
        
        valueCountIntegerList \
            = sorted \
                (tempSeries \
                     .unique() \
                     .tolist())

        valueCountIntegerList \
            .insert(0, 0)

        log_function \
            .DebugReturnObjectWriteObject \
                (valueCountIntegerList)
        
        
        return tempSeries, valueCountIntegerList
    
    except:
        
        log_subroutine \
            .PrintAndLogWriteText \
                (f'The function, ReturnColumnSeriesAndSortedValueCountList, '
                 + f'in source file, {CONSTANT_LOCAL_FILE_NAME}, '
                 + f'was unable to return the column Series and sorted value count List.')
        
        return None


# In[4]:


def SetFeaturesInteger \
        (featuresInteger):
    
    local_constant.featuresInteger \
        = featuresInteger


# In[5]:


def ReturnFeaturesInteger():
        
        return local_constant.featuresInteger


# In[6]:


def ReturnBinnedDataFrameForOneColumn \
        (tempDataFrame,
         tempSeries,
         tempCountIntegerList,
         columnNameString,
         countInteger):
    
    try:
        
        if countInteger == 0:
            
            return tempDataFrame
        
        elif len(tempCountIntegerList) == 3 and countInteger != max(tempCountIntegerList):
            
            return tempDataFrame
        
        elif countInteger == max(tempCountIntegerList):
            
            tempDataFrame \
                .drop \
                    ([columnNameString], 
                     axis = 1, 
                     inplace = True)
            
            return tempDataFrame
        
        
        typesToReplaceList \
            = list \
                (tempSeries \
                     [tempSeries < countInteger].index)

        log_function \
            .DebugReturnObjectWriteObject \
                (typesToReplaceList)
        
        
        for typesToReplace in typesToReplaceList:
    
            tempDataFrame[columnNameString] \
                = tempDataFrame[columnNameString] \
                    .replace \
                        (typesToReplace, 'Other')
    
        log_function \
            .DebugReturnObjectWriteObject \
                (tempDataFrame)
        
        return tempDataFrame
    
    except:
        
        log_subroutine \
            .PrintAndLogWriteText \
                (f'The function, ReturnBinnedDataFrameFromOneColumn, '
                 + f'in source file, {CONSTANT_LOCAL_FILE_NAME}, '
                 + f'was unable to return a binned DataFrame from one column.')
        
        return None


# In[7]:


def ReturnBinnedDataFrame \
        (inputDataFrame,
         inputSeriesList,
         inputCountIntegerListList,
         columnNameStringList,
         countIntegerList):
    
    try:
        
        tempDataFrame \
            = inputDataFrame.copy()
        
        for index, inputSeries in enumerate(inputSeriesList):
            
            tempDataFrame \
                = ReturnBinnedDataFrameForOneColumn \
                    (tempDataFrame,
                     inputSeries,
                     inputCountIntegerListList[index],
                     columnNameStringList[index],
                     countIntegerList[index])
            
        return tempDataFrame
        
    except:
        
        log_subroutine \
            .PrintAndLogWriteText \
                (f'The function, ReturnBinnedDataFrame, '
                 + f'in source file, {CONSTANT_LOCAL_FILE_NAME}, '
                 + f'was unable to return a binned DataFrame.')
        
        return None


# In[8]:


def ReturnNeuralNetworkXYParameters \
        (inputDataFrame,
         outcomeColumnNameString):
        
    try:

        dummiesDataFrame \
            = pd.get_dummies \
                (inputDataFrame)
                             
        yNumpyArray \
            = dummiesDataFrame \
                [outcomeColumnNameString] \
                    .values

        XNumpyArray \
            = dummiesDataFrame \
                .drop \
                    ([outcomeColumnNameString],
                     axis = 1) \
                .values

        XTrainNumpyArray, \
        XTestNumpyArray, \
        yTrainNumpyArray, \
        yTestNumpyArray \
            = train_test_split \
                (XNumpyArray, 
                 yNumpyArray, 
                 random_state = 9)
        
        currentStandardScalar \
            = StandardScaler()

        XStandardScalar \
            = currentStandardScalar \
                .fit \
                    (XTrainNumpyArray)

        XTrainScaledNumpyArray \
            = XStandardScalar \
                .transform \
                    (XTrainNumpyArray)

        XTestScaledNumpyArray \
            = XStandardScalar \
                .transform \
                    (XTestNumpyArray)
        
        return \
            XTrainScaledNumpyArray, \
            XTestScaledNumpyArray, \
            yTrainNumpyArray, \
            yTestNumpyArray
        
    except:
        
        log_subroutine \
            .PrintAndLogWriteText \
                (f'The function, ReturnNeuralNetworkXYParameters, '
                 + f'in source file, {CONSTANT_LOCAL_FILE_NAME}, '
                 + f"was unable to return a the neural network's X-Y parameters.")
        
        return None, None, None, None


# In[9]:


def ReturnNeuralNetworkModel \
        (hp):
    
    try:
        
        inputFeaturesInteger \
            = ReturnFeaturesInteger()
            
        neuralNetSequential \
            = tf.keras.models.Sequential()

        activationChoice \
            = hp.Choice \
                ('activation',
                 ['relu',
                  'sigmoid',
                  'tanh',
                  'leaky_relu',
                  'hard_sigmoid'])
    
        neuralNetSequential \
            .add \
                (tf.keras.layers.Dense \
                    (units \
                         = hp.Int \
                             ('first_units',
                              min_value = 1,
                              max_value = 100,
                              step = 1), 
                     activation = activationChoice,
                     input_dim = inputFeaturesInteger))

        for index in range(hp.Int('num_layers', 1, 5)):
            
            neuralNetSequential \
                .add \
                    (tf.keras.layers.Dense \
                         (units \
                              = hp.Int \
                                  ('units_' + str(index),
                                   min_value = 1,
                                   max_value = 100,
                                   step = 1),
                                   activation = activationChoice))

        neuralNetSequential \
            .add \
                (tf.keras.layers.Dense \
                     (units = 1, 
                      activation = 'sigmoid'))
        
        learningRateFloat \
            = hp.Choice \
                ('learning_rate', 
                 values = [1e-2, 1e-3, 1e-4])

        neuralNetSequential \
            .compile \
                (loss = 'binary_crossentropy', 
                 optimizer = tf.keras.optimizers.Adam(learning_rate = learningRateFloat),
                 metrics = ['accuracy'])

    
        return neuralNetSequential
        
    except:
        
        log_subroutine \
            .PrintAndLogWriteText \
                (f'The function, ReturnNeuralNetworkModel, '
                 + f'in source file, {CONSTANT_LOCAL_FILE_NAME}, '
                 + f'was unable to return a neural network model.')
        
        return None


# In[10]:


def ReturnBestModelDictionary \
        (XTrainScaledNumpyArray, \
         XTestScaledNumpyArray, \
         yTrainNumpyArray, \
         yTestNumpyArray,
         objectiveString,
         maxEpochsInteger,
         hyperbandIterationsInteger,
         numberOfTopModelsInteger):
    
    try:

        tunerHyperband \
            = kt.Hyperband \
                (ReturnNeuralNetworkModel,
                 objective = objectiveString,
                 max_epochs = maxEpochsInteger,
                 hyperband_iterations = hyperbandIterationsInteger,
                 overwrite = True)

        tunerHyperband.search \
            (XTrainScaledNumpyArray,
             yTrainNumpyArray,
             epochs = maxEpochsInteger,
             validation_data = (XTestScaledNumpyArray, yTestNumpyArray))
        
        topHyperparametersList \
            = tunerHyperband \
                .get_best_hyperparameters \
                    (numberOfTopModelsInteger)
        
        topModelList \
            = tunerHyperband \
                .get_best_models \
                    (numberOfTopModelsInteger)
        
        log_subroutine \
                .PrintAndLogWriteText \
                    (f'\nTop {numberOfTopModelsInteger} models:\n\n')
        
        for model in topModelList:
            
            modelLossFloat, \
            modelAccuracyFloat \
                = model.evaluate \
                    (XTestScaledNumpyArray,
                     yTestNumpyArray,
                     verbose = 2)
            
            log_subroutine \
                .PrintAndLogWriteText \
                    (f'\nModel Loss: {round(modelLossFloat * 100, 2)}%, ' 
                     + f'Model Accuracy: {round(modelAccuracyFloat*100, 2)}%')
        
        bestHyperparameters \
            = tunerHyperBand \
                .get_best_hyperparameters()[0]
        
        bestModelNeuralNetSequential \
            = tunerHyperBand \
                .get_best_models(numberOfTopModelsInteger)[0]
        
        bestModelLossFloat, \
        bestModelAccuracyFloat \
            = bestModelNeuralNetSequential \
                .evaluate \
                    (XTestScaledNumpyArray,
                     yTestNumpyArray,
                     verbose = 2)

        log_subroutine \
            .PrintAndLogWriteText \
                (f'\nModel Loss: {round(bestModelLossFloat * 100, 2)}%, ' 
                 + f'Model Accuracy: {round(bestModelAccuracyFloat*100, 2)}%\n'
                 + bestHyperparameters.values)
        
        

        tempDictionary \
            = {'accuracy': bestModelLossFloat,
               'loss': bestModelAccuracyFloat,
               'hyperparameters': bestHyperparameters.values,
               'duration': '',
               'count_list': [],
               'max_count_list': []}
        
        
        return tempDictionary
        
    except:
        
        log_subroutine \
            .PrintAndLogWriteText \
                (f'The function, ReturnBestModelDictionary, '
                 + f'in source file, {CONSTANT_LOCAL_FILE_NAME}, '
                 + f'was unable to return Dictionary with the best model parameters.')
        
        return None


# In[ ]:




