'''
-*- coding: utf-8 -*-

This module contains data preprocessing and sampling utilities.

Author: Shuyang Li
License: GNU GPLv3
'''

import os
import gc
import pandas as pd
import numpy as np
from datetime import datetime

USER_ITEM_COLS = {
    'allrecipes': ['user_id', 'recipe_id'],
    'movielens': ['user_id', 'item_id']
}

def print_basic_stats(df, user_col, item_col):
    '''
    Prints basic summary stats for a DF:
        # users
        # items
        # interactions
        sparsity %
        avg. interactions/user
        avg. interactions/item
        memory usage

    Arguments:
        df {pd.DataFrame} -- DataFrame with user and item interactions recorded
        user_col {str} -- user ID column name
        item_col {str} -- item ID column name
    '''
    gc.collect()

    n_items = df[item_col].nunique()
    n_users = df[user_col].nunique()
    print('{} Users interacted with {} items {} times ({:.4f}% sparsity, {:.3f} actions/user, {:.3f} actions/item)'.format(
        n_users, n_items, len(df), 100.0 * (1.0 - float(len(df))/(n_items * n_users)),
        len(df)/n_users, len(df)/n_items
    ))

    # Get base dataset memory usage
    print('')
    print(df.info(memory_usage='deep'))
    print('')

def get_base_data(dataset):
    '''
    Retrieve base data DF (unprocessed)
    
    Arguments:
        dataset {str} -- Name of dataset (e.g. `allrecipes`, `movielens`)
    
    Raises:
        Exception -- Specified dataset not supported
    
    Returns:
        pd.DataFrame -- DataFrame with the following columns:
            user_col
            item_col
            'date'
        str -- Name of original user ID column
        str -- Name of original item ID column
    '''
    # Sanity check
    if dataset not in USER_ITEM_COLS:
        raise Exception('Requested dataset {} not supported. Use one from: {}'.format(
            dataset, set(USER_ITEM_COLS.keys())
        ))
    
    start = datetime.now()
    user_col, item_col = USER_ITEM_COLS[dataset]
    if dataset == 'allrecipes':
        # Get original msgpack
        df = pd.read_msgpack('allrecipes_uri_enriched.msgpack')
    elif dataset == 'movielens':
        movie_dir = os.path.join('datasets', 'ml-1m')
        interactions_loc = os.path.join(movie_dir, 'interactions.msgpack')

        # Get original msgpack
        df = pd.read_msgpack(interactions_loc)

        # Process date
        df['date'] = pd.to_datetime(df['timestamp'].apply(datetime.utcfromtimestamp))
    
    df = df.drop(columns=[c for c in df.columns if c not in {user_col, item_col, 'date'}])
    print('{} - Retrieved base DF for {}'.format(
        datetime.now() - start, dataset
    ))

    return df, user_col, item_col

def process_temporal_columns(df):
    '''
    Gets the following temporal columns:
        year
        month
        day_of_week
        day_of_year (adjusted for leap year, Feb 29 -> 0)
    
    Arguments:
        df {pd.DataFrame} -- DataFrame with a 'date' column
    
    Returns:
        pd.DataFrame -- DataFrame with additional information
    '''
    start = datetime.now()
    df['year'] = df['date'].dt.year.astype(int)
    df['month'] = df['date'].dt.month.astype(int)
    df['day_of_week'] = df['date'].dt.dayofweek.astype(int)

    # Adjust day of year -> move february 29 from day 60 to day 0
    df['day_of_year'] = df['date'].dt.dayofyear.astype(int)
    df['is_leap_year'] = df['date'].dt.is_leap_year.astype(int)
    df.loc[(df['day_of_year'] == 60) & (df['is_leap_year'] == True), 'day_of_year'] = 0
    df.loc[(df['day_of_year'] > 60) & (df['is_leap_year'] == True), 'day_of_year'] -= 1
    
    df = df.drop(columns=['is_leap_year'])
    print('{} - Added proper temporal columns to df'.format(
        datetime.now() - start
    ))
    return df

def kcore_interaction_stats(df, user_col, item_col, core):
    '''
    Performs k-core on a graph. Preserves all users with at least k interactions and all items with at least k interactions.
    
    Arguments:
        df {pd.DataFrame} -- DataFrame with user and item interactions recorded
        user_col {str} -- user ID column name
        item_col {str} -- item ID column name
        core {int} -- cores
    
    Returns:
        invalid_users -- set of IDs of users with fewer than k interactions
        invalid_items -- set of IDs of items with fewer than k interactions
    '''
    n_items = df[item_col].nunique()
    n_users = df[user_col].nunique()
    user_degrees = df.groupby([user_col])[item_col].count()
    item_degrees = df.groupby([item_col])[user_col].count()
    invalid_users = set(user_degrees[user_degrees < core].index)
    invalid_items = set(item_degrees[item_degrees < core].index)
    print('Removing {}/{} users ({:.2f} %) and {}/{} items ({:.2f} %) from {} total interactions ({:.5f}% Sparsity)'.format(
        len(invalid_users), n_users, 100 * len(invalid_users) / n_users,
        len(invalid_items), n_items, 100 * len(invalid_items) / n_items,
        len(df), 100 * (1 - len(df) / (n_items * n_users))
    ))
    return invalid_users, invalid_items

def kcore(df, user_col, item_col, core):
    '''
    Performs k-core on a graph. Preserves all users with at least k interactions and all items with at least k interactions.
    
    Arguments:
        df {pd.DataFrame} -- DataFrame with user and item interactions recorded
        user_col {str} -- user ID column name
        item_col {str} -- item ID column name
        core {int} -- cores
    
    Returns:
        pd.DataFrame -- k-core graph DF
    '''
    start = datetime.now()
    iters = 0
    while True:
        invalid_users, invalid_items, = kcore_interaction_stats(df, user_col, item_col, core)
        iters += 1
        if len(invalid_users) == 0 and len(invalid_items) == 0:
            print('{} - Done: {}-core decomposition after {} iterations'.format(
                datetime.now() - start, core, iters
            ))
            break
        
        # Remove invalid users and items
        df = df[~df[user_col].isin(invalid_users)]
        df = df[~df[item_col].isin(invalid_items)]
    
    return df

def map_user_items(df, user_col, item_col):
    '''
    Maps user/item IDs to integer IDs.
    
    Arguments:
        df {pd.DataFrame} -- DataFrame with user and item interactions recorded
        user_col {str} -- user ID column name
        item_col {str} -- item ID column name
    
    Returns:
        pd.DataFrame -- DataFrame with mapped [0, n_users) users and [0, n_items) items, as well as 
                        prior item for each item/user combo.
    '''
    start = datetime.now()

    # Get all unique reviewers and items across training and test set
    unique_users = np.array(list(df[user_col].unique()))
    unique_items = np.array(list(df[item_col].unique()))
    n_items = len(unique_items)
    n_users = len(unique_users)

    # MAP USERS AND ITEMS TO INT
    df = df.drop(columns=['u', 'i'], errors='ignore')
    user_map = pd.DataFrame(list(zip(unique_users, np.arange(n_users))), columns=[user_col, 'u'])
    item_map = pd.DataFrame(list(zip(unique_items, np.arange(n_items))), columns=[item_col, 'i'])
    df = pd.merge(df, user_map, on=user_col)
    df = pd.merge(df, item_map, on=item_col)
    df['u'] = df['u'].astype(np.int32)
    df['i'] = df['i'].astype(np.int32)
    print('{} - Mapped u-i indices'.format(datetime.now() - start))

    # Create 'prior item'
    df = df.sort_values(['date'])
    df['prior'] = df.groupby(['u'])['i'].shift(1)
    df = df.dropna(subset=['prior'])
    df['prior'] = df['prior'].astype(int)
    print('{} - Created "prior" column'.format(datetime.now() - start))

    return df

def get_processed_df(dataset, cores, overwrite=False):
    '''
    Processes a DataFrame to include the following columns:
        original user ID
        original item ID
        'u' - mapped integer user ID
        'prior' - mapped integer item ID of previous item (in per-user item sequence)
        'i' - mapped integer item ID
        'date'
        'year'
        'month'
        'day_of_year'
        'day_of_week'
    
    Applies the following preprocessing steps:
        K-core
        Get temporal columns
        Map user/item IDs to contiguous integer series
    
    Arguments:
        dataset {str} -- Name of a supported dataset (e.g. `allrecipes`, `gk`, `movielens`)
        cores {int} -- Minimum # of interactions per user and item
    
    Keyword Arguments:
        overwrite {bool} -- Whether to re-process data (default: {False})
    
    Returns:
        pd.DataFrame -- Processed DataFrame
        str -- Original user ID column name
        str -- Original item ID column name
    '''
    # Sanity check
    if dataset not in USER_ITEM_COLS:
        raise Exception('Requested dataset {} not supported. Use one from: {}'.format(
            dataset, set(USER_ITEM_COLS.keys())
        ))
    
    dataset_name = '{}_{}.msgpack'.format(dataset, cores)
    user_col, item_col = USER_ITEM_COLS[dataset]

    # If preprocessed and stored ahead of time, retrieve it
    if os.path.exists(dataset_name) and not overwrite:
        start = datetime.now()
        df = pd.read_msgpack(dataset_name)
        print('{} - Retrieved preprocessed data for {}, {}-cores'.format(
            datetime.now() - start, dataset, cores
        ))
        print_basic_stats(df, user_col, item_col)
        return df, user_col, item_col
    
    # Get basic data
    df, user_col, item_col = get_base_data(dataset)

    # Get temporal columns
    df = process_temporal_columns(df)

    # K-core
    df = kcore(df, user_col, item_col, cores)

    # Mapping
    df = map_user_items(df, user_col, item_col)

    # Get stats
    gc.collect()
    print_basic_stats(df, user_col, item_col)

    # Store DF
    start = datetime.now()
    df.to_msgpack(dataset_name)
    print('{} - Processed and stored dataset to {}'.format(
        datetime.now() - start, dataset_name
    ))

    return df, user_col, item_col