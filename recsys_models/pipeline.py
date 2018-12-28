'''
-*- coding: utf-8 -*-

Model training pipelines

Author: Shuyang Li
License: GNU GPLv3
'''
import tensorflow as tf
from datetime import datetime
from recsys_models.data.sampling import uniform_sample_from_df

def train_model(session, model, train_df, validation_mat, test_mat,
                n_iterations=2500, batch_size=512,
                min_epochs=10, max_epochs=200, stopping_threshold=1e-5,
                **sampling_kwargs):
    '''
    Perform mini-batch optimization with the given model on the provided data.
    
    Arguments:
        session {tf.Session} -- TensorFlow session instance
        model {RecSysModel} -- RecSysModel instance
        train_df {pd.DataFrame} -- DF of user-item interactions for training data
        validation_mat {np.array} -- matrix / 2d array of validation ranking tuples
        test_mat {np.array} -- matrix / 2d array of test ranking tuples
    
    Keyword Arguments:
        n_iterations {int} -- Number of batches per epoch (Default: {2500})
        batch_size {int} -- Number of training examples per mini-batch (Default: {512})
        min_epochs {int} -- Train model for at least this many epochs (Default: {10})
        max_epochs {int} -- Maximum number of epochs for which to train model (Default: {200})
        stopping_threshold {float} -- Stop the training if validation AUC change falls below this value(Default: {1e-5})
    
    Returns:
        RecSysModel -- Trained model object
        float -- Training AUC
        float -- Validation AUC
        float -- Test AUC
    '''
    start = datetime.now()
    
    # Get initial validation AUC
    prior_auc = model.evaluate_auc(session, validation_mat)
    test_auc = model.evaluate_auc(session, test_mat)
    print('{} - Prior: {:.5f} Validation AUC, {:.5f} Testing AUC'.format(
        datetime.now() - start, prior_auc, test_auc
    ))

    # Epochs of training
    epoch_num = 0
    for epoch_num in range(max_epochs):

        # Make epoch training batch
        training_batch = uniform_sample_from_df(
            train_df, n_iterations * batch_size, **sampling_kwargs
        )

        # Train the model
        epoch_loss = model.train_epoch(session, training_batch, n_iterations, batch_size)

        # Get the full training/validation AUCs
        train_auc = model.evaluate_auc(session, training_batch)
        validation_auc = model.evaluate_auc(session, validation_mat)

        # Compute change in validation AUC for stoppage
        delta_auc = validation_auc - prior_auc
        prior_auc = validation_auc
        print('[{} - Epoch {}] {:.5f} Loss, {:.5f} Training AUC, {:.5f} Validation AUC ({:.5f} Change)'.format(
            datetime.now() - start, epoch_num + 1, epoch_loss, train_auc, validation_auc, delta_auc
        ))

        # Stopping condition (give it a few epochs to find its bearings)
        if epoch_num > min_epochs and delta_auc < stopping_threshold:
            break
    
    # Evaluate the final trained model
    test_auc = model.evaluate_auc(session, test_mat)
    print('[{} - Epoch {}] - STOPPED. Final test AUC: {:.5f}'.format(
        datetime.now() - start, epoch_num + 1, test_auc
    ))
    
    return model, train_auc, validation_auc, test_auc