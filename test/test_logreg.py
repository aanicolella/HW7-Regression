"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
from regression import (utils, logreg)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

def test_prediction():
	
	# load data, split to test and validation set
	X_train, X_val, y_train, y_val = utils.loadDataset(
		features = [
        'Penicillin V Potassium 500 MG',
        'Computed tomography of chest and abdomen',
        'Plain chest X-ray (procedure)', 
        'Low Density Lipoprotein Cholesterol',
        'Creatinine',
        'AGE_DIAGNOSIS'],
		split_percent=0.6,
		split_seed= 42
	)
	X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)
	
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)
	X_test = sc.transform(X_test)

	# train/fit model
	lr = logreg.LogisticRegressor(num_feats=X_train.shape[1],
							   max_iter=250)
	lr.train_model(X_train, y_train, X_val, y_val)

	# train scikitlearn LR model and make predictions for comparison to our model
	sk_lr = LogisticRegression(solver='saga',
                                   max_iter=250,
                                   random_state=42
                                   )
	sk_lr.fit(X_train, y_train)
	# Manually align weights/intercept with that of our model
	sk_lr.coef_  = lr.W[:-1].reshape(1, -1)
	sk_lr.intercept_ = np.array([lr.W[-1]])

	# for our LR method, make predictions on test set, convert probabilities to binary prediction
	y_hat = lr.make_prediction(X_test)
	y_hat = np.where(y_hat > 0.5, 1, 0)
	# make predictions for sklearn method
	sk_yhat = sk_lr.predict(X_test)

	# check that predictions from both models are equivalent
	assert np.array_equal(y_hat, sk_yhat)

	# get accuracy for our method
	y_hat_acc = accuracy_score(y_hat, y_test)
	sk_acc = accuracy_score(y_test, sk_yhat)
	# check that accuracies are similar
	assert np.isclose(y_hat_acc, sk_acc), 'Accuracy of model is insufficient'


def test_loss_function():
	# load data, only need train for this 
	X_train, _, y_train, _ = utils.loadDataset(
		features = [
        'Penicillin V Potassium 500 MG',
        'Computed tomography of chest and abdomen',
        'Plain chest X-ray (procedure)', 
        'Low Density Lipoprotein Cholesterol',
        'Creatinine',
        'AGE_DIAGNOSIS'],
		split_percent=0.8,
		split_seed= 42
	)

	# create random y_pred values for testing loss functions 
	y_pred = np.random.random(size=y_train.shape)

	# calculate loss values using our loss_function
	lr = logreg.LogisticRegressor(num_feats=X_train.shape[1])
	lr_loss = lr.loss_function(y_train, y_pred)

	# get loss values from scikitlearn's log_loss function
	sk_loss = log_loss(y_train, y_pred)

	# check that losses are similar
	assert np.isclose(lr_loss, sk_loss), 'Loss values are insufficient'

def test_gradient():
	# load data, only need small subset for testing gradient on one variable
	_, X_val, _, y_val = utils.loadDataset(
		features = ['Penicillin V Potassium 500 MG',
            'Computed tomography of chest and abdomen',
            'Plain chest X-ray (procedure)',
            'Low Density Lipoprotein Cholesterol',
            'Creatinine',
            'AGE_DIAGNOSIS'],
		split_percent=0.95,
		split_seed= 42
	)

	# split subset into training and testing
	X_train, X_test, y_train, y_test = train_test_split(X_val, y_val, test_size=0.025, random_state=42)
	
	# data scaling
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)

	print(f'y_test: {y_test}')

	# create instance of our LR model, train, calculate gradients
	lr = logreg.LogisticRegressor(num_feats=X_val.shape[1], max_iter=500)
	lr.train_model(X_train, y_train, X_test, y_test)

	# manually add bias column to X_test
	X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
	lr_grad = lr.calculate_gradient(y_test, X_test)	

	# calculate gradient and save as array for comparison to our model's calculation
	grad_byHand = np.zeros_like(lr.W)
	ep = 1e-5

	for i in range(len(lr.W)):
		W_plusEp = lr.W.copy()
		W_plusEp[i] += ep
		W_minEp = lr.W.copy()
		W_minEp[i] -= ep
		
		lr.W = W_plusEp
		loss_plusEp = lr.loss_function(y_test, lr.make_prediction(X_test))

		lr.W = W_minEp
		loss_minEp = lr.loss_function(y_test, lr.make_prediction(X_test))

		grad_byHand[i] = (loss_plusEp - loss_minEp) / (2 * ep)

	# check that gradients calculated using our code and by hand are equivalent
	assert np.allclose(lr_grad, grad_byHand, atol= 1e-3), 'Gradient calculation is insufficient.'



def test_training():
	# load data, split to train and validation
	X_train, X_val, y_train, y_val = utils.loadDataset(
		features = [
        'Penicillin V Potassium 500 MG',
        'Computed tomography of chest and abdomen',
        'Plain chest X-ray (procedure)', 
        'Low Density Lipoprotein Cholesterol',
        'Creatinine',
        'AGE_DIAGNOSIS'],
		split_percent=0.8,
		split_seed= 42
	)
	# create lr instance and get initial weights
	lr = logreg.LogisticRegressor(num_feats=X_train.shape[1], max_iter=250)

	# data scaling
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)

	# create lr instance and get initial weights
	lr = logreg.LogisticRegressor(num_feats=X_train.shape[1],
							   max_iter=250)
	init_W = lr.W.copy()

	# train model and get post training weights
	lr.train_model(X_train, y_train, X_val, y_val)
	trained_W = lr.W.copy()

	# check that weights have changed (been updated) during training
	assert np.array_equal(init_W, trained_W) == False, 'Weights have not been properly updated during training.'