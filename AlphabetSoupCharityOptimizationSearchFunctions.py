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


#*******************************************************************************************
 #
 #  Function Name:  ReturnColumnSeriesAndSortedValueCountList
 #
 #  Function Description:
 #      This function receives a DataFrame and column name and returns the column 
 #      as a Series and a sorted list of unique values from the Series.
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  DataFrame
 #          inputDataFrame
 #                          The parameter is the input DataFrame.
 #  String
 #          columnNameString
 #                          This parameter is the DataFrame column name.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  12/2/2023           Initial Development                         N. James George
 #
 #******************************************************************************************/

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


#*******************************************************************************************
 #
 #  Function Name:  SetFeaturesInteger
 #
 #  Function Description:
 #      This function sets the global variable, featuresInteger.
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  Integer
 #          featuresInteger
 #                          The parameter is the new value for the global variable.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  12/2/2023           Initial Development                         N. James George
 #
 #******************************************************************************************/

def SetFeaturesInteger \
        (featuresInteger):
    
    local_constant.featuresInteger \
        = featuresInteger


# In[5]:


#*******************************************************************************************
 #
 #  Function Name:  ReturnFeaturesInteger
 #
 #  Function Description:
 #      This function returns the global variable, featuresInteger.
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  Integer
 #          featuresInteger
 #                          The parameter is the new value for the global variable.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  12/2/2023           Initial Development                         N. James George
 #
 #******************************************************************************************/

def ReturnFeaturesInteger():
        
        return local_constant.featuresInteger


# In[6]:


#*******************************************************************************************
 #
 #  Function Name:  ReturnBinnedDataFrameForOneColumn
 #
 #  Function Description:
 #      This function returns one binned column for a input DataFrame.
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  DataFrame
 #          inputDataFrame
 #                          The parameter is the input DataFrame.
 #  DataFrame
 #          inputDataFrame
 #                          The parameter is the input Series for the DataFrame column.
 #  List
 #          inputCountIntegerList
 #                          The parameter is the sorted List of unique values for the DataFrame
 #                          column.    
 #  String
 #          columnNameString
 #                          This parameter is the DataFrame column name.
 #  Integer
 #          countInteger
 #                          This parameter is the binning limit.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  12/2/2023           Initial Development                         N. James George
 #
 #******************************************************************************************/

def ReturnBinnedDataFrameForOneColumn \
        (inputDataFrame,
         inputSeries,
         inputCountIntegerList,
         columnNameString,
         countInteger):
    
    try:
        
        if countInteger == 0:
            
            return inputDataFrame
        
        elif len(inputCountIntegerList) == 3 and countInteger != max(inputCountIntegerList):
            
            return inputDataFrame
        
        elif countInteger == max(inputCountIntegerList):
            
            inputDataFrame \
                .drop \
                    ([columnNameString], 
                     axis = 1, 
                     inplace = True)
            
            return inputDataFrame
        
        
        typesToReplaceList \
            = list \
                (inputSeries \
                     [inputSeries < countInteger].index)

        log_function \
            .DebugReturnObjectWriteObject \
                (typesToReplaceList)
        
        
        for typesToReplace in typesToReplaceList:
    
            tempDataFrame[columnNameString] \
                = inputDataFrame[columnNameString] \
                    .replace \
                        (typesToReplace, 'Other')
    
        log_function \
            .DebugReturnObjectWriteObject \
                (inputDataFrame)
        
        return inputDataFrame
    
    except:
        
        log_subroutine \
            .PrintAndLogWriteText \
                (f'The function, ReturnBinnedDataFrameFromOneColumn, '
                 + f'in source file, {CONSTANT_LOCAL_FILE_NAME}, '
                 + f'was unable to return a binned DataFrame from one column.')
        
        return None


# In[7]:


#*******************************************************************************************
 #
 #  Function Name:  ReturnBinnedDataFrame
 #
 #  Function Description:
 #      This function returns one binned column for a input DataFrame.
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  DataFrame
 #          inputDataFrame
 #                          The parameter is the input DataFrame.
 #  List
 #          inputSeriesList
 #                          The parameter is the List of input Series for the DataFrame columns.
 #  List of List
 #          inputCountIntegerListList
 #                          The parameter is the sorted List of unique values for the DataFrame
 #                          column.    
 #  String
 #          columnNameStringList
 #                          This parameter is the List of DataFrame column names.
 #  Integer
 #          countIntegerList
 #                          This parameter is the List of binning limits.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  12/2/2023           Initial Development                         N. James George
 #
 #******************************************************************************************/

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


#*******************************************************************************************
 #
 #  Function Name:  ReturnNeuralNetworkXYParameters
 #
 #  Function Description:
 #      This function returns one training and testing X-Y parameters for a neural network.
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  DataFrame
 #          inputDataFrame
 #                          The parameter is the input DataFrame.
 #  String
 #          outcomeColumnNameString
 #                          The parameter is the columns name for the y-variable.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  12/2/2023           Initial Development                         N. James George
 #
 #******************************************************************************************/

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
                 random_state = 21)
        
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


#*******************************************************************************************
 #
 #  Function Name:  ReturnNeuralNetworkModel
 #
 #  Function Description:
 #      This function returns one neural network model for analysis.
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  Hyperband
 #          hp
 #                          The parameter is the input Tensorflow Hyperband object.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  12/2/2023           Initial Development                         N. James George
 #
 #******************************************************************************************/

def ReturnNeuralNetworkModel \
        (hp):
    
    try:
        
        inputFeaturesInteger \
            = ReturnFeaturesInteger()
            
        neuralNetSequentialModel \
            = tf.keras.models.Sequential()

        activationChoice \
            = hp.Choice \
                ('activation',
                 ['relu',
                  'sigmoid',
                  'tanh',
                  'leaky_relu',
                  'hard_sigmoid'])
    
        neuralNetSequentialModel \
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
            
            neuralNetSequentialModel \
                .add \
                    (tf.keras.layers.Dense \
                         (units \
                              = hp.Int \
                                  ('units_' + str(index),
                                   min_value = 1,
                                   max_value = 100,
                                   step = 1),
                                   activation = activationChoice))

        neuralNetSequentialModel \
            .add \
                (tf.keras.layers.Dense \
                     (units = 1, 
                      activation = 'sigmoid'))
        
        learningRateFloat \
            = hp.Float \
                    ('learning_rate', 
                     min_value = 1e-4, 
                     max_value = 1e-2, 
                     sampling = 'linear')

        neuralNetSequentialModel \
            .compile \
                (loss = 'binary_crossentropy', 
                 optimizer = tf.keras.optimizers.Adam(learning_rate = learningRateFloat),
                 metrics = ['accuracy'])

    
        return neuralNetSequentialModel
        
    except:
        
        log_subroutine \
            .PrintAndLogWriteText \
                (f'The function, ReturnNeuralNetworkModel, '
                 + f'in source file, {CONSTANT_LOCAL_FILE_NAME}, '
                 + f'was unable to return a neural network model.')
        
        return None


# In[10]:


#*******************************************************************************************
 #
 #  Function Name:  ReturnBestModelDictionary
 #
 #  Function Description:
 #      This function returns one training and testing X-Y parameters for a neural network.
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  Numpy Array
 #          XTrainScaledNumpyArray
 #                          The parameter is the training array for the x-variable.
 #  Numpy Array
 #          XTestScaledNumpyArray
 #                          The parameter is the testing array for the x-variable.
 #  Numpy Array
 #          yTrainScaledNumpyArray
 #                          The parameter is the training array for the y-variable.
 #  Numpy Array
 #          yTestScaledNumpyArray
 #                          The parameter is the testing array for the y-variable.
 #  String
 #          objectiveString
 #                          The parameter is the objective of the analysis 
 #                          (i.e., val_accuracy).
 #  Integer
 #          maxEpochsInteger
 #                          The parameter is the maximum number of epochs for the compiling 
 #                          phase.
 #  Integer
 #          hyperbandIterationsInteger
 #                          The parameter is the number of iterations for the Hyperband
 #                          function.
 #  Integer
 #          numberOfTopModelsInteger
 #                          The parameter is the number of top final models from the 
 #                          analysis.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  12/2/2023           Initial Development                         N. James George
 #
 #******************************************************************************************/

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




