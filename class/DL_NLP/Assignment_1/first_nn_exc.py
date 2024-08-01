"""
-----------------------------------------------------------------------------
A simple two layers neural network for classification task. Some part of this 
excercise taken from Andrej Karpathy (https://karpathy.ai/)
-----------------------------------------------------------------------------
AUTHOR: Soumitra Samanta (soumitra.samanta@gm.rkmvu.ac.in)
-----------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    'sigmoid_func',
    'FirstNN',
]
    

def sigmoid_func(z):
    
     return 1/(1 + np.exp(-z)) 

class FirstNN(object):
  """
  A simple two-layer fully-connected neural network for a classification (C classes) task.
  
  Network architechture: Input (D -dims) -> M hidden neurons -> Sigmoid activation function -> C output neurons -> Softmax -> Cross-entropy loss 

  """

  def __init__(self, input_dims, num_nodes_lr1, num_classes, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_dims: The dimension D of the input data.
    - num_nodes_lr1: The number of neurons H in the hidden layer.
    - num_classes: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_dims, num_nodes_lr1)
    self.params['b1'] = np.zeros(num_nodes_lr1)
    self.params['W2'] = std * np.random.randn(num_nodes_lr1, num_classes)
    self.params['b2'] = np.zeros(num_classes)

  def loss(self, X, Y=None, reg=0.0):
    """
    Compute the loss and gradients.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - Y: Vector of training labels. Y[i] is the label for X[i], and each Y[i] is
      an integer in the range 0 <= Y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization parameter.

    Returns:
    If Y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If Y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    z1=np.dot(X,W1) + b1
    o1=sigmoid_func(z1)
    z2=np.dot(o1,W2) + b2
    scores=z2
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then we're done
    if Y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2 (if reg !=0). Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    o2=np.exp(z2)/np.sum(np.exp(z2),axis=1,keepdims=True)
    # print(o2.shape)
    for k in range(N):
      for j in range(o2.shape[1]):
        

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    
        
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, Y, X_val, Y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train the neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - Y: A numpy array f shape (N,) giving training labels; Y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - Y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    
    self.params['learning_rate'] = learning_rate
    self.params['learning_rate_decay'] = learning_rate_decay
    self.params['reg'] = reg
    self.params['num_iters'] = num_iters
    self.params['batch_size'] = batch_size

    num_train_data = X.shape[0]
    iterations_per_epoch = max(num_train_data / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
      X_batch = None
      Y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      mask = np.random.choice(num_train_data, batch_size)
      X_batch = X[mask]
      Y_batch = Y[mask]  
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, Y=Y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      
        
        
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
#         print('iteration %d / %d: loss %f' % (it, num_iters, loss))
        print('iteration {} / {}: loss {}' .format(it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == Y_batch).mean()
        val_acc = (self.predict(X_val) == Y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)


    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained network to predict labels for data points. For each data 
    point we predict scores for each of the C classes, and assign each data 
    point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - Y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, Y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    Y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    
    
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return Y_pred


