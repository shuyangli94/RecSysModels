'''
-*- coding: utf-8 -*-

This module contains implementations for various implicit feedback recommender systems models

Author: Shuyang Li
License: GNU GPLv3
'''

from abc import ABCMeta, abstractmethod
from datetime import datetime
import os
import pickle
import pandas as pd
import numpy as np

class RecSysModel(metaclass=ABCMeta):
    """
    Encapsulating class for Recommender System models.
    """
    def __init__(self):
        self.model_id = 'TEST'
        self.loss = None
        self.training_optimizer = None
        self.p_uij = None

    ######################################
    # Model Saving/Serialization
    ######################################
    
    @property
    def params(self):
        '''
        Returns trained weights for all parameters in the model.
        
        Arguments:
            session {tf.Session} -- TensorFlow session object
        
        Returns:
            object -- Model parameter 1
            ...
        '''
        raise NotImplementedError
    
    @abstractmethod
    def get_weights(self, session):
        '''
        Returns trained weights for all parameters in the model.
        
        Arguments:
            session {tf.Session} -- TensorFlow session object
        
        Returns:
            np.ndarray -- Weights for parameter 1
            ...
        '''
        raise NotImplementedError

    def save(self, session, loc, suffix=''):
        '''
        Saves model parameters and weights to the following files, respectively:
            <loc>
                <model ID>
                    params.pkl
                    weights.pkl
        
        Arguments:
            session {tf.Session} -- TensorFlow session
            loc {str} -- Parent folder to store the item
        
        Keyword Arguments:
            suffix {str} -- Optimal string to append to model storage. (Default: {''})
        '''
        start = datetime.now()

        # Create the model folder if it doesn't exist already
        model_folder = os.path.join(loc, '{}{}'.format(self.model_id, suffix))
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)

        # Save parameters
        params_file_loc = os.path.join(model_folder, 'params.pkl')
        with open(params_file_loc, 'wb') as model_params_file:
            pickle.dump(
                self.params,
                model_params_file,
                protocol=pickle.HIGHEST_PROTOCOL
            )
        print('{} - Saved parameters to {}'.format(
            datetime.now() - start, params_file_loc
        ))

        # Save weights
        weights_file_loc = os.path.join(model_folder, 'weights.pkl')
        with open(weights_file_loc, 'wb') as model_weights_file:
            pickle.dump(
                self.get_weights(session),
                model_weights_file,
                protocol=pickle.HIGHEST_PROTOCOL
            )
        print('{} - Saved weights to {}'.format(
            datetime.now() - start, weights_file_loc
        ))

    @classmethod
    def load(cls, loc):
        '''
        Given a model folder with saved parameters and weights, reconstruct the model
        
        Arguments:
            loc {str} -- Location of the saved model folder, containing:
                            params.pkl
                            weights.pkl
        
        Returns:
            RecSysModel -- Model with loaded pretrained weights.
        '''
        # Load parameters
        with open(os.path.join(loc, 'params.pkl'), 'rb') as model_params_file:
            params_list = list(pickle.load(model_params_file))
            if params_list is None:
                params_list = []
        
        # Load weights
        with open(os.path.join(loc, 'weights.pkl'), 'rb') as model_weights_file:
            weights_list = list(pickle.load(model_weights_file))
            if weights_list is None:
                weights_list = []
        
        return cls(*(params_list + weights_list))
    
    ######################################
    # Training and Evaluation
    ######################################

    @abstractmethod
    def _session_run(self, session, input_batch, *args):
        '''
        Computes graph variables based on inputs.
        
        Arguments:
            session {tf.Session} -- TF Session
            input_batch {np.ndarray} -- 2d array or matrix
        
        Arbitrary Arguments:
            *args {tf.Variable} -- TF variables to be computed
        
        Returns:
            list -- TF Variable values
        '''
        raise NotImplementedError
    
    @abstractmethod
    def debug(self, session, input_batch):
        '''
        Debugger - indicates where variables are NaN / 

        Arguments:
            session {tf.Session} -- TF Session
            input_batch {np.ndarray} -- 2d array or matrix

        Raises:
            Exception
        '''
        raise NotImplementedError

    def train_batch(self, session, input_batch):
        '''
        Training with a single batch

        Arguments:
            session {tf.Session} -- TF Session
            input_batch {np.ndarray} -- 2d array or matrix
        
        Returns:
            float -- Batch loss
        '''
        batch_loss, _ = self._session_run(session, input_batch, self.loss, self.training_optimizer)

        # Identify errors in batch loss
        if np.isnan(batch_loss):
            self.debug(session, input_batch)

        return batch_loss

    def train_epoch(self, session, input_matrix, n_iterations, batch_size):
        '''
        Trains for a single epoch
        
        Arguments:
            session {tf.Session} -- TF Session
            input_matrix {np.ndarray} -- 2d array or matrix containing all of the training data
                                         for that epoch
            n_iterations {int} -- Number of batches per epoch
            batch_size {int} -- Number of training examples per batch
        
        Returns:
            float -- Epoch loss
        '''
        epoch_loss = 0.0

        # Train on each batch
        for iter_num in range(1, n_iterations + 1):
            input_batch = input_matrix[batch_size * (iter_num-1) : batch_size * iter_num, :]
            batch_loss = self.train_batch(session, input_batch)
            epoch_loss += batch_loss
        
        epoch_loss = epoch_loss / float(n_iterations)
        return epoch_loss

    def evaluate_auc(self, session, input_data):
        '''
        Evaluate the rankings for testing/validation data
        
        Arguments:
            session {tf.Session} -- TF Session
            input_data {np.ndarray} -- 2d array or matrix
        
        Returns:
            float -- AUC for the input data
        '''
        # Get predictions
        ranking_predictions = self._session_run(session, input_data, self.p_uij)[0]

        # This is the magic - it's cheaper to plug it into a pandas DF and then
        # groupby-mean-mean to do mean-of-means on x_uij -> AUC
        pred_df = pd.DataFrame(input_data[:, :3], columns=['u', 'i', 'j'])
        pred_df['ranking'] = ranking_predictions
        pred_df['prediction'] = pred_df['ranking'] > 0
        auc = pred_df[['u', 'prediction']].groupby(['u']).mean()['prediction'].mean()
        return auc

def pop_rec(train_df, eval_df):
    '''
    PopRec model: For a triplet (u, i, j) of observed item i and unobserved item j, rank first the
    item that was most popular in the training data.
    
    Arguments:
        train_df {pd.DataFrame} -- DF of training user-item interactions
        eval_df {pd.DataFrame} -- DF of evaluation user-item interactions
    
    Returns:
        float -- PopRec AUC
    '''
    # Get popularity of each item in the training data
    train_popularities = train_df[['u', 'i']].groupby(['i'])['u'].count().to_dict()

    # For each (u, i, j) triplet, positive prediction is whether i is more popular than j.
    predictions = [train_popularities.get(i, 0) > train_popularities.get(j, 0) for \
                    i, j in zip(eval_df['i'], eval_df['j'])]
    
    # AUC = mean of per-user AUC
    auc = eval_df[['u']].assign(yhat=predictions).groupby(['u'])['yhat'].mean().mean()
    return auc
