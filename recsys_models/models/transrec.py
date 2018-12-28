'''
-*- coding: utf-8 -*-

TensorFlow implementation of Translational Recommendations for implicit feedback

Source paper(s):
Translation-based recommendation
    He, et al. 2017

Author: Shuyang Li
License: GNU GPLv3
'''

import tensorflow as tf
from recsys_models.models import RecSysModel
import pandas as pd
import numpy as np

class TransRec(RecSysModel):
    '''
    Translation-Based Recommender System

    Input format:
        (u, p, i, j) - user u, prior item p, observed item i, unobserved item j
    
    Intuition for pairwise ranking for implicit feedback:
        Users will rank observed items above unobserved items
        We want to maximize the log likelihood of the batch rankings
    
    Intuition for translation embedding:
        Items are embedded as points in the shared embedding space
        Users embedded as translational vectors in the shared embedding space
        Prior Item (point) + User (vector) -> Next Item (point location), and thus:
            P(i|u, p) ~ L2 loss of gamma_i - (gamma_p + gamma_u)

    Variables
        gamma_u -> user translation vector embedding
        gamma_p -> prior item point embedding
        gamma_i -> observed item point embedding
        gamma_j -> unobserved item point embedding
        beta_i -> observed item bias
        beta_j -> unobserved item bias
        global_u -> global user translation bias
    
    Weights to be optimized are on the order of:
        k * (n_items + n_users + 1) + n_items
    
    Weights:
        U_emb (n_users x k) -- User translation embeddings
        I_emb (n_items x k) -- Item point embeddings
        Bi_emb (n_items x 1) -- Item biases
        Global_u (k x 1) -- Global user translation vector basis (alpha-analogue)
    
    Optimization Criterion:
        Maximize sum of log-likelihoods of rankings:
            ln(sigmoid(p_ui - p_uj))
        Where
            p_ui -> P(i|u), prop. to ||gamma_u + gamma_p - gamma_i||
            p_uj -> P(j|u), prop. to ||gamma_u + gamma_p - gamma_j||
    
    Regularization:
        embeddings (gamma_u, gamma_p, gamma_i, gamma_j)
        biases (beta_i, beta_j, global_u)
    '''
    def __init__(
        self, n_users, n_items, k=2, lambda_emb=1e-6, lambda_bias=1e-6,
        opt_type=tf.contrib.opt.LazyAdamOptimizer,
        opt_args=dict(),
        U_emb_weights=None,
        I_emb_weights=None,
        Bi_emb_weights=None,
        Global_u_weights=None):
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
            U_emb_weights {np.ndarray} -- Initial weights for latent user factors (default: {None})
            I_emb_weights {np.ndarray} -- Initial weights for latent item factors (default: {None})
            Bi_emb_weights {np.ndarray} -- Initial weights for item biases (default: {None})
            Global_u_weights {np.ndarray} -- Initial values for global translation vector prior (default: {None})
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
        self.model_id = 'transrec_{}k_{}l2_{}l2bias'.format(
            self.k, self.lambda_emb, self.lambda_bias
        )

        # Initialized variable weights (None-type checks because of ambiguous truth values for arr)
        U_emb_init = U_emb_weights if U_emb_weights is not None else \
            tf.zeros([n_users, k], dtype=tf.float32)
        I_emb_init = I_emb_weights if I_emb_weights is not None else \
            tf.nn.l2_normalize(
                tf.random_normal([n_items, k], stddev=(1.0/(k**0.5)), dtype=tf.float32),
                axis=1
            )
        Bi_emb_init = Bi_emb_weights if Bi_emb_weights is not None else \
            tf.zeros([n_items, 1], dtype=tf.float32)
        Global_u_init = Global_u_weights if Global_u_weights is not None else \
            tf.nn.l2_normalize(
                tf.random_normal([1, k], stddev=(1.0/(k**0.5)), dtype=tf.float32),
                axis=1
            )
        
        # (Batch) Placeholders
        self.u = tf.placeholder(tf.int32, [None])  # User ID
        self.p = tf.placeholder(tf.int32, [None])  # Prior item ID
        self.i = tf.placeholder(tf.int32, [None])  # Observed item ID
        self.j = tf.placeholder(tf.int32, [None])  # Unobserved item ID

        # Variables - normalize to unit ball to mitigate curse of dimensionality (Lin et al. 2015)
        self.U_emb = tf.Variable(           # User vector embedding matrix
            initial_value=U_emb_init,
            trainable=True
        )
        self.I_emb = tf.Variable(           # Item point embedding matrix
            initial_value=I_emb_init,
            trainable=True
        )
        self.Bi_emb = tf.Variable(          # Item bias embedding vector
            initial_value=Bi_emb_init,
            trainable=True
        )
        self.Global_u = tf.Variable(        # Global user translation prior
            initial_value=Global_u_init,
            trainable=True
        )

        # Batch Embeddings
        self.u_trans_vec = self.Global_u + tf.nn.embedding_lookup(self.U_emb, self.u)
        self.p_points = tf.nn.embedding_lookup(self.I_emb, self.p)
        self.i_points = tf.nn.embedding_lookup(self.I_emb, self.i)
        self.j_points = tf.nn.embedding_lookup(self.I_emb, self.j)
        self.i_biases = tf.nn.embedding_lookup(self.Bi_emb, self.i)
        self.j_biases = tf.nn.embedding_lookup(self.Bi_emb, self.j)

        # Likelihoods
        self.p_ui = self.i_biases - tf.sqrt(
            tf.reduce_mean(
                tf.square(self.p_points + self.u_trans_vec - self.i_points),
                1, keepdims=True
            ) + 1e-8
        )
        self.p_uj = self.j_biases - tf.sqrt(
            tf.reduce_mean(
                tf.square(self.p_points + self.u_trans_vec - self.j_points),
                1, keepdims=True
            ) + 1e-8
        )
        self.p_uij = self.p_ui - self.p_uj
        # Add epsilon for validity at 0
        self.log_likelihood = tf.log(tf.sigmoid(self.p_uij) + 1e-8)

        # Regularization - Factorization terms
        self.l2_emb = self.lambda_emb * tf.add_n([
            tf.nn.l2_loss(self.u_trans_vec),
            tf.nn.l2_loss(self.p_points),
            tf.nn.l2_loss(self.i_points),
            tf.nn.l2_loss(self.j_points),
            tf.nn.l2_loss(self.Global_u)
        ])

        # Regularization - Bias terms
        self.l2_bias = self.lambda_bias * tf.add_n([
            tf.nn.l2_loss(self.i_biases),
            tf.nn.l2_loss(self.j_biases)
        ])
        
        # Loss
        self.loss = self.l2_emb + self.l2_bias - tf.reduce_sum(self.log_likelihood)

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
            np.ndarray -- Trained global translation vector prior weights
        '''
        # User embedding matrix
        U_emb_weights = session.run(self.U_emb)

        # Item embedding matrix
        I_emb_weights = session.run(self.I_emb)

        # Item bias embedding vector
        Bi_emb_weights = session.run(self.Bi_emb)

        # Global translation vector prior
        Global_u_weights = session.run(self.Global_u)

        return U_emb_weights, I_emb_weights, Bi_emb_weights, Global_u_weights

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
                self.p: input_batch[:, 1],
                self.i: input_batch[:, 2],
                self.j: input_batch[:, 3]
            }
        )

    def debug(self, session, input_batch):
        '''
        Debugger - indicates where variables are NaN / 

        Arguments:
            session {tf.Session} -- TF Session
            input_batch {np.ndarray} -- 2d array or matrix with the following column order:
                user ID (u)
                prior item ID (p)
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
