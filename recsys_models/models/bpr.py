'''
-*- coding: utf-8 -*-

TensorFlow implementation of Bayesian Personalized Ranking for implicit feedback,
with underlying Matrix Factorization model

Source paper(s):
BPR: Bayesian Personalized Ranking from Implicit Feedback
    Rendle, et al. 2009

Author: Shuyang Li
License: GNU GPLv3
'''

import tensorflow as tf
from recsys_models.models import RecSysModel
import pandas as pd
import numpy as np

class BPR_MF(RecSysModel):
    '''
    Bayesian Personalized Ranking with Matrix Factorization

    Input format:
        (u, i, j) - user u, observed item i, unobserved item j
    
    Intuition for pairwise ranking for implicit feedback:
        Users will rank observed items above unobserved items
        We want to maximize the log likelihood of the batch rankings
    
    Variables
        gamma_u -> user embedding
        gamma_i -> observed item embedding
        gamma_j -> unobserved item embedding
        beta_i -> observed item bias 
        beta_j -> unobserved item bias
    
    Weights to be optimized are on the order of:
        k * (n_items + n_users) + n_items
    
    Weights:
        U_mf (n_users x k) -- User latent factor matrix
        I_mf (n_items x k) -- Item latent factor matrix
        Bi_emb (n_items x 1) -- Item biases
    
    Optimization Criterion:
        Maximize sum of log-likelihoods of rankings:
            ln(sigmoid(p_ui - p_uj))
        Where
            p_ui -> P(i|u), prop. to <gamma_u, gamma_i>
            p_uj -> P(j|u), prop. to <gamma_u, gamma_j>
    
    Regularization:
        embeddings (gamma_u, gamma_i, gamma_j)
        biases (beta_i, beta_j)
    '''
    def __init__(
        self, n_users, n_items, k=2, lambda_emb=1e-6, lambda_bias=1e-6,
        opt_type=tf.contrib.opt.LazyAdamOptimizer,
        opt_args=dict(),
        U_mf_weights=None,
        I_mf_weights=None,
        Bi_mf_weights=None):
        '''
        Arguments:
            n_users {int} -- Number of users
            n_items {int} -- Number of items
        
        Keyword Arguments:
            k {int} -- Latent dimensionality (default: {2})
            lambda_emb {float} -- Embedding regularization rate (default: {1e-6})
            lambda_bias {float} -- Bias term regularization rate (default: {1e-6})
            opt_type {tf.train.Optimizer} -- TF optimizer class (default: {tf.contrib.opt.LazyAdamOptimizer})
            opt_args {dict} -- Dictionary of arguments for the TF optimizer class
            U_mf_weights {np.ndarray} -- Initial weights for latent user factors (default: {None})
            I_mf_weights {np.ndarray} -- Initial weights for latent item factors (default: {None})
            Bi_mf_weights {np.ndarray} -- Initial weights for item biases (default: {None})
        '''
        # Parameters
        self.n_users = n_users                      # Number of users
        self.n_items = n_items                      # Number of items
        self.k = k                                  # Latent dimensionality
        self.lambda_emb = lambda_emb                # Regularization rate (embeddings)
        self.lambda_bias = lambda_bias              # Regularization rate (bias)
        self.optimizer = opt_type(**opt_args)       # Optimizer
        self.opt_type = opt_type                    # Optimizer class
        self.opt_args = opt_args                    # Optimizer arguments

        # Model ID
        self.model_id = 'bpr-mf_{}k_{}l2_{}l2bias'.format(
            self.k, self.lambda_emb, self.lambda_bias
        )

        # Initialized variable weights (None-type checks because of ambiguous truth values for arr)
        U_mf_init = U_mf_weights if U_mf_weights is not None else \
            tf.nn.l2_normalize(
                tf.random_normal([n_users, k], stddev=(1.0/(k**0.5)), dtype=tf.float32),
                axis=1
            )
        I_mf_init = I_mf_weights if I_mf_weights is not None else \
            tf.nn.l2_normalize(
                tf.random_normal([n_items, k], stddev=(1.0/(k**0.5)), dtype=tf.float32),
                axis=1
            )
        Bi_mf_init = Bi_mf_weights if Bi_mf_weights is not None else \
            tf.zeros([n_items, 1], dtype=tf.float32)
        
        # (Batch) Placeholders
        self.u = tf.placeholder(tf.int32, [None])  # User ID
        self.i = tf.placeholder(tf.int32, [None])  # Observed item ID
        self.j = tf.placeholder(tf.int32, [None])  # Unobserved item ID

        # Variables - normalize to unit ball to mitigate curse of dimensionality (Lin et al. 2015)
        self.U_mf = tf.Variable(            # User embedding matrix
            initial_value=U_mf_init,
            trainable=True
        )
        self.I_mf = tf.Variable(            # Item embedding matrix
            initial_value=I_mf_init,
            trainable=True
        )
        self.Bi_mf = tf.Variable(           # Item bias embedding vector
            initial_value=Bi_mf_init,
            trainable=True
        )

        # Batch Embeddings
        self.u_embeddings = tf.nn.embedding_lookup(self.U_mf, self.u)
        self.i_embeddings = tf.nn.embedding_lookup(self.I_mf, self.i)
        self.j_embeddings = tf.nn.embedding_lookup(self.I_mf, self.j)
        self.i_biases = tf.nn.embedding_lookup(self.Bi_mf, self.i)
        self.j_biases = tf.nn.embedding_lookup(self.Bi_mf, self.j)

        # Likelihoods
        self.p_ui = tf.reduce_sum(
            tf.multiply(self.u_embeddings, self.i_embeddings), 1, keepdims=True
            ) + self.i_biases
        self.p_uj = tf.reduce_sum(
            tf.multiply(self.u_embeddings, self.j_embeddings), 1, keepdims=True
            ) + self.j_biases
        self.p_uij = self.p_ui - self.p_uj
        # Add epsilon for validity at 0
        self.log_likelihood = tf.log(tf.sigmoid(self.p_uij) + 1e-8)

        # Regularization - Factorization terms
        self.l2_emb = self.lambda_emb * tf.add_n([
            tf.nn.l2_loss(self.u_embeddings),
            tf.nn.l2_loss(self.i_embeddings),
            tf.nn.l2_loss(self.j_embeddings)
        ])

        # Regularization - Bias terms
        self.l2_bias = self.lambda_bias * tf.add_n([
            tf.nn.l2_loss(self.i_biases),
            tf.nn.l2_loss(self.j_biases)
        ])
        
        # Loss
        self.loss = self.l2_emb + self.l2_bias - tf.reduce_mean(self.log_likelihood)

        # Training optimizer
        self.training_optimizer = self.optimizer.minimize(self.loss)

    ######################################
    # Model Saving/Serialization
    ######################################

    @property
    def params(self):
        return [
            self.n_users,           # Number of users
            self.n_items,           # Number of items
            self.k,                 # Latent dimensionality
            self.lambda_emb,        # Regularization rate (embeddings)
            self.lambda_bias,       # Regularization rate (bias)
            self.opt_type,          # Optimizer class
            self.opt_args,          # Optimizer arguments
        ]
    
    def get_weights(self, session):
        '''
        Returns trained weights for all parameters in the model.
        
        Arguments:
            session {tf.Session} -- TensorFlow session object
        
        Returns:
            np.ndarray -- Trained user embedding matrix weights
            np.ndarray -- Trained item embedding matrix weights
            np.ndarray -- Trained item bias weights
        '''
        # User embedding matrix
        U_mf_weights = session.run(self.U_mf)

        # Item embedding matrix
        I_mf_weights = session.run(self.I_mf)

        # Item bias embedding vector
        Bi_mf_weights = session.run(self.Bi_mf)

        return U_mf_weights, I_mf_weights, Bi_mf_weights

    ######################################
    # Training and Evaluation
    ######################################

    def _session_run(self, session, input_batch, *args):
        '''
        Computes graph variables based on inputs.
        
        Arguments:
            session {tf.Session} -- TF Session
            input_batch {np.ndarray} -- 2d array or matrix with the following column order:
                user ID (u)
                observed item ID (i)
                unobserved item ID (j)
        
        Arbitrary Arguments:
            *args {tf.Variable} -- TF variables to be computed
        
        Returns:
            list -- TF Variable values
        '''
        return session.run(
            args,
            feed_dict={
                self.u: input_batch[:, 0],
                self.i: input_batch[:, 1],
                self.j: input_batch[:, 2]
            }
        )

    def debug(self, session, input_batch):
        '''
        Debugger - indicates where variables are NaN / 

        Arguments:
            session {tf.Session} -- TF Session
            input_batch {np.ndarray} -- 2d array or matrix with the following column order:
                user ID (u)
                observed item ID (i)
                unobserved item ID (j)

        Raises:
            Exception
        '''
        # Common intermediaries
        p_ui, p_uj, p_uij, log_likelihood, l2_emb, l2_bias = \
        self._session_run(
            session, input_batch,
            self.p_ui,
            self.p_uj,
            self.p_uij,
            self.log_likelihood,
            self.l2_emb,
            self.l2_bias
        )

        # Identify problematic i-preferences
        nan_pui_ix = np.argwhere(np.isnan(p_ui))
        if nan_pui_ix.size > 0:
            print('ERROR - NaN p_ui at {} from batch data {}'.format(
                nan_pui_ix, input_batch[nan_pui_ix, :]
            ))

        # Identify problematic j-preferences
        nan_puj_ix = np.argwhere(np.isnan(p_uj))
        if nan_puj_ix.size > 0:
            print('ERROR - NaN p_uj at {} from batch data {}'.format(
                nan_puj_ix, input_batch[nan_puj_ix, :]
            ))

        # Identify problematic p_uij = p_ui - p_uj
        nan_puij_ix = np.argwhere(np.isnan(p_uij))
        if nan_puij_ix.size > 0:
            print('ERROR - NaN p_uij at {} from batch data {}'.format(
                nan_puij_ix, input_batch[nan_puij_ix, :]
            ))

        # Identify problematic Log Likelihood log(sig(p_uij))
        nan_LL_ix = np.argwhere(np.isnan(log_likelihood))
        if nan_LL_ix.size > 0:
            print('ERROR - NaN Log Likelihood at {} from batch data {}'.format(
                nan_LL_ix, input_batch[nan_LL_ix, :]
            ))

        # Identify problematic L2 regularization term (embeddings)
        nan_l2e_ix = np.argwhere(np.isnan(l2_emb))
        if nan_l2e_ix.size > 0:
            print('ERROR - NaN Embedding Reg term at {} from batch data {}'.format(
                nan_l2e_ix, input_batch[nan_l2e_ix, :]
            ))

        # Identify problematic L2 regularization term (bias)
        nan_l2b_ix = np.argwhere(np.isnan(l2_bias))
        if nan_l2b_ix.size > 0:
            print('ERROR - NaN Bias Reg term at {} from batch data {}'.format(
                nan_l2b_ix, input_batch[nan_l2b_ix, :]
            ))

        raise Exception('ERROR IN BATCH')
