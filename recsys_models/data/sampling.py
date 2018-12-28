'''
-*- coding: utf-8 -*-

Sampling and train/test/validation splitting utilities

Author: Shuyang Li
License: GNU GPLv3
'''

import os
import gc
import pandas as pd
import numpy as np
from datetime import datetime
import random

def sample_unobserved(df, user_interactions_dict, n_items, size=500000, use_original_actions=False):
    '''
    Samples unobserved items for each (user, item) interaction pair.
    Creates <size> pairwise comparison tuples (user, observed item, unobserved item) where the 
    observed items are drawn from the provided DF.
    
    Arguments:
        df {pd.DataFrame} -- DataFrame of user-item interactions
        user_interactions_dict {dict} -- Dictionary of user : observed items for that user
        n_items {int} -- Number of unique items
    
    Keyword Arguments:
        size {int} -- Desired number of rows in the output DataFrame (Default: {500000})
        use_original_actions {bool} -- If True, `size` argument is ignored and the original interaction
                                    rows are used. (Default: {False})
    
    Returns:
        pd.DataFrame -- DataFrame with 'j' unobserved items
    '''
    if use_original_actions:
        output_df = df.copy()
    else:
        output_df = df.sample(n=size, replace=True).reset_index(drop=True)
    js = []

    # Iterating through a series or list is WAY faster than iterrows on a DF!!
    for u in output_df['u']:
        j = random.randint(0, n_items - 1)
        while j in user_interactions_dict[u]:
            j = random.randint(0, n_items - 1)
        js.append(j)

    output_df['j'] = js
    return output_df

def uniform_sample_from_df(
        base_df, size, n_items, items_by_user,
        sample_columns=['u', 'i', 'j'], column_order=['u', 'i', 'j'],
        item_properties_df=None):
    '''
    Uniformly samples user-item-unobserved interactions from a provided DataFrame.
    This function is more general than `sample_unobserved` and allows for column selection,
    ordering, and attaching item properties/features.
    
    Arguments:
        base_df {dict} -- User-item interactions DF from which we generate positive samples
        size {int} -- Number of interactions to sample
        n_items {int} -- Number of total items
        items_by_user {dict} -- Mapping of user -> IDs of items they have interacted with
    
    Keyword Arguments:
        sample_columns {list} -- Which columns to sample from base DF (default: {['u', 'i']})
        column_order {list} -- Order of columns in matrix output (default: {['u', 'i', 'j']})
        item_properties_df {pd.DataFrame} -- DF containing attributes/features per item
                                            (default: {None})

    Returns:
        np.ndarray -- 2D array of sampled rows
    '''
    # Uniformly sample positive+negative interactions
    batch_df = sample_unobserved(base_df[sample_columns], items_by_user, n_items, size)

    # Fill out item properties if we are using features
    if item_properties_df is not None:
        j_properties = item_properties_df.loc[batch_df['j'].values]
        for col in j_properties.columns:
            batch_df[col] = j_properties[col].values
    
    return batch_df[column_order].values

def get_user_interactions_df(base_df):
    '''
    Gets a DataFrame of items and # interactions indexed by user
    
    Arguments:
        base_df {pd.DataFrame} -- DataFrame of user-item interactions
    
    Returns:
        pd.DataFrame -- DataFrame indexed by user, with the following columns:
            count -- # of interactions made by this user
            items -- Set of items that the user has interacted with
    '''
    by_user_df = base_df.groupby(['u']).agg({
        'i': [
            'count',
            lambda x: set(x)
        ]
    })
    by_user_df.columns = ['count', 'items']
    return by_user_df

def train_test_validation_split(df, size):
    '''
    Generates training, testing, and validation DataFrames from a provided interactions DF

    First, we generate holdout sets:
        Latest interaction/user -> test holdout
        Second-to-last interaction/user -> validation holdout
    
    Test and validation datasets are created thusly:
        (u, i) observed item interactions are drawn from the respective holdout sets
        (j), the unobserved item for user u, is randomly sampled
    
    Arguments:
        df {pd.DataFrame} -- DataFrame of all user-item interactions
        size {int} -- Number of rows in each evaluation set (test/validation)
    
    Returns:
        pd.DataFrame -- Training DF, WITHOUT unobserved items (j)
        pd.DataFrame -- Validation DF, WITH unobserved items already sampled (j)
        pd.DataFrame -- Testing DF, WITH unobserved items already sampled (j)
        pd.DataFrame -- Per-user interactions DF, with the following columns:
                            'count' -- # of interactions by that user
                            'items' -- Set of IDs for items that the user has interacted with
    '''
    start = datetime.now()
    n_items = df['i'].nunique()

    # Create the user -> n_interactions, items mapping DF
    by_user_df = get_user_interactions_df(df)
    by_user_dict = by_user_df['items'].to_dict()
    print('{} - Created full user : item, interaction mappings df and user : items dict'.format(
        datetime.now() - start
    ))

    # Create holdout sets and training DF
    holdout = df.groupby(['u'], as_index=False).tail(2)
    holdout_test = holdout.groupby(['u'], as_index=False).tail(1)
    train_df = df.loc[~df.index.isin(holdout.index)].copy()
    holdout_validation = holdout.loc[~holdout.index.isin(holdout_test.index)]
    print('{} - Created holdout sets of most recent interactions/user'.format(
        datetime.now() - start
    ))

    # Create validation DF
    validation_df = sample_unobserved(
        holdout_validation,
        by_user_dict,
        n_items,
        size)
    print('{} - Created bootstrap validation DF of size {}'.format(
        datetime.now() - start, len(validation_df)
    ))

    # Create test DF
    test_df = sample_unobserved(
        holdout_test,
        by_user_dict,
        n_items,
        size)
    print('{} - Created bootstrap testing DF of size {}'.format(
        datetime.now() - start, len(test_df)
    ))

    return train_df, validation_df, test_df, by_user_df
