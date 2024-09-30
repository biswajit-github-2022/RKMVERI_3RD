"""
-----------------------------------------------------------------------------
A simple two layers neural network for classification task. Some parts of this 
excercise taken from https://cs231n.github.io/assignments2017/assignment1/
-----------------------------------------------------------------------------
AUTHOR: Soumitra Samanta (soumitra.samanta@gm.rkmvu.ac.in)
-----------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm

__all__ = [
    'sigmoid_func',
    'FirstNN',
]
    

def sigmoid_func(z):
    
    """
    Sigmoid function and its operate on each element of the inut vector z
    """
    
    return 1/(1 + np.exp(-z)) 

class FirstNN(object):
    """
    A simple two-layer fully-connected neural network for a classification (C classes) task.

    Network architechture: Input (D -dims) -> M hidden neurons -> Sigmoid activation function -> C output neurons -> Softmax -> Cross-entropy loss 

    """

    def __init__(self, input_dims, num_nodes_lr1, num_classes, param_inits='small_std',std=1e-4):
        
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, M)
        b1: First layer biases; has shape (M,)
        W2: Second layer weights; has shape (M, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
            - input_dims: The dimension D of the input data.
            - num_nodes_lr1: The number of neurons M in the hidden layer.
            - num_classes: The number of classes C.
            - std: Scaling factor for weights initialization

        """
        
        self.params = {}

        self.params['b1'] = np.zeros(num_nodes_lr1)
        self.params['b2'] = np.zeros(num_classes)

        self.params['V_b1'] = np.zeros(num_nodes_lr1)
        self.params['V_b2'] = np.zeros(num_classes)

        if param_inits=='small_std':
            self.params['W1']=std * np.random.randn(input_dims,num_nodes_lr1)
            self.params['W2']=std * np.random.randn(num_nodes_lr1,num_classes)

        elif param_inits=='ninn_std':
            self.params['W1']=np.random.normal(loc=0.0,scale=1/np.sqrt(input_dims),size=(input_dims,num_nodes_lr1))
            self.params['W2']=np.random.normal(loc=0.0,scale=1/np.sqrt(num_nodes_lr1),size=(num_nodes_lr1,num_classes))
            

        elif param_inits=='Xavier':
            print(param_inits)
            self.params['W1']=np.random.normal(loc=0.0,scale=np.sqrt(2/input_dims+num_nodes_lr1),size=(input_dims,num_nodes_lr1))
            self.params['W2']=np.random.normal(loc=0.0,scale=np.sqrt(2/num_nodes_lr1+num_classes),size=(num_nodes_lr1,num_classes))
            self.best_params=copy.deepcopy(self.params)






        # self.params['W1'] = std * np.random.randn(input_dims, num_nodes_lr1)
        # self.params['b1'] = np.zeros(num_nodes_lr1)
        # self.params['W2'] = std * np.random.randn(num_nodes_lr1, num_classes)
        # self.params['b2'] = np.zeros(num_classes)
        self.params['V_W1']=np.zeros((input_dims,num_nodes_lr1))
        self.params['V_W2']=np.zeros((num_nodes_lr1,num_classes))

        self.best_params = copy.deepcopy(self.params)
        
            
    
    def forword(self, X):
        
        """
        Compute the scores (forward pass).

        Inputs:
            - X (N, D): Input data, X[i, :] is the i-th training sample.

        Outputs:
            - prob_scores (N, C): Probability scores,  prob_scores[i, c] is the 
            score for class c on input X[i].
        """
        
        # Forward pass
        prob_scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        Z_1 = np.dot(X, self.params['W1']) + self.params['b1']
        self.O_1 = sigmoid_func(Z_1)
        scores = np.dot(self.O_1, self.params['W2']) + self.params['b2']
        
        exp_scores = np.exp(scores)
        prob_scores = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        
        return prob_scores
        
        
    def loss(self, Y, prob_scores):
        
        """
        Compute loss (cross-entropy).
        
        Inputs:
            - Y (N): Labels of the X, Y[i] is the label of X[i, :].
            - prob_scores (N, C): Probability scores from forword pass. 
            prob_scores[i, c] is the score for class c on input X[i].
         
        Outputs:
            - loss: A scalar value.
        """
        loss = None
        #############################################################################
        # TODO:  Use the cross-entropy loss.                                        #
        #############################################################################
        loss = 0.0
        N = Y.shape[0]
        loss -= np.sum(np.log(prob_scores[(range(N), Y)]))
        loss /= N
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        
        return loss
        
        
    def backword(self, X, Y, prob_scores):
        """
        Compute the gradients (backword pass).
        
        Input:
            - X (N, D): Input data, X[i, :] is the i-th training sample.
            - Y (N): Labels of the X, Y[i] is the label of X[i, :].
            - prob_scores (N, C): Probability scores from forword pass, prob_scores[i, c] 
            is the score for class c on input X[i].
            
        Output:
            - grads (dictionary): A dictionary holds the gradients of nework's weights. 
        """
        
        # Backword pass (calculate gradient)
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        N, D = X.shape
        prob_score_neg_y = copy.deepcopy(prob_scores)
        prob_score_neg_y[(range(N),Y)] -= 1# (N, C)
        
        grads['W2'] = np.dot(self.O_1.T, prob_score_neg_y)/N #(M, C)
        grads['b2'] = np.sum(prob_scores, axis=0)/N #(C)
        
        dZ_1 = np.dot(prob_score_neg_y, self.params['W2'].T)*(self.O_1*(1 - self.O_1)) #(N, M)
        
        grads['W1'] = np.dot(X.T, dZ_1)/N #(D,M)
        grads['b1'] = np.sum(dZ_1, axis=0)/N #(M)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        
        return grads
    
    
    def optimizer(self, grads,update_rule='gd'):
        
        """
        Update parameters using gradient decent
        
        Inputs: 
            - grads (dictionary): A dictionary holds the gradients of nework's weights.
            
        Outputs:
            - None
        """
        
        #########################################################################
        # TODO: Use the gradients in the grads dictionary to update the         #
        # parameters of the network (stored in the dictionary self.params)      #
        # using gradient descent. You'll need to use the gradients stored in    #
        # the grads dictionary defined above.                                   #
        #########################################################################

        if update_rule=='gd':    
            self.params['W1'] -= self.learning_rate*grads['W1']
            self.params['b1'] -= self.learning_rate*grads['b1']
            self.params['W2'] -= self.learning_rate*grads['W2']
            self.params['b2'] -= self.learning_rate*grads['b2']
        elif update_rule =='m_gd':
            self.params['V_W1'] =self.beta_moment*self.params['V_W1'] - self.learning_rate*grads['W1']
            self.params['W1']+=self.params['V_W1']
            self.params['V_b1'] =self.beta_moment*self.params['V_b1'] - self.learning_rate*grads['b1']
            self.params['b1']+=self.params['V_b1']
            self.params['V_W2'] =self.beta_moment*self.params['V_W2'] - self.learning_rate*grads['W2']
            self.params['W2']+=self.params['V_W2']
            self.params['V_b2'] =self.beta_moment*self.params['V_b2'] - self.learning_rate*grads['b2']
            self.params['b2']+=self.params['V_b2']
            self
        elif update_rule =='m_gd':
            pass
        #########################################################################
        #                             END OF YOUR CODE                          #
        #########################################################################
        
        
        
    def train(self, X, Y, X_val, Y_val,
              num_iters=100, 
              num_epoch=None,
              batch_size=200, 
              learning_rate=1e-3, 
              verbose=False,
              update_rule='gd',
              beta_moment=1e-1
             ):
        
        """
        Train the neural network using stochastic gradient descent.

        Inputs:
            - X (N, D): Training data, X[i, :] is a i-th training sample.
            - Y (N): Training data labels, Y[i] = c means that X[i, :] has label c, where 0 <= c < C.
            - X_val (N_val, D): Validation data, X_val[i, :] is a i-th training sample.
            - Y_val (N_val): Validation data labels, Y_val[i] = c means that X_val[i, :] has label c, where 0 <= c < C.
            - num_iters: Number of steps for optimization of networ's weights.
            - num_epoch: Number of epochs for optimization of networ's weights.
            - batch_size: Number of training examples to use per step.
            - learning_rate: Learning rate for optimization.
            - verbose (boolean): If true print progress during optimization.
        """

        self.num_iters = num_iters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        self.update_rule=update_rule
        self.beta_moment=beta_moment

        loss_history_batch = []
        loss_history_epoch = []
        train_acc_history = []
        val_acc_history = []
        train_acc = 0
        val_acc = 0
        best_val_acc = 0

        num_train_data = X.shape[0]
        
        # Use SGD to optimize the parameters in self.model 
        
        # SGD vertion-1:
        if num_epoch == None:
            iterations_per_epoch = round(max(num_train_data / batch_size, 1))
            if verbose:
                process_bar = tqdm(range(num_iters))
            else:
                process_bar = range(num_iters)
            epoch_train_loss = 0
            for it in process_bar:
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
                # Forword pass
                prob_scores = self.forword(X_batch)

                # Loss
                loss_batch = self.loss(Y_batch, prob_scores)
                loss_history_batch.append(loss_batch)
                epoch_train_loss += loss_batch

                # Calculate gradients
                grads_batch = self.backword(X_batch, Y_batch, prob_scores)

                # Update the parameters
                self.optimizer(grads_batch,update_rule)

                # Every epoch, check train and val accuracy and record the best weights
                if it % iterations_per_epoch == 0:
                    epoch_train_loss /= iterations_per_epoch    
                    loss_history_epoch.append(epoch_train_loss)
                    epoch_train_loss = 0
                    # Check accuracy
                    train_acc = 100*(self.predict(X) == Y).mean()
                    val_acc = 100*(self.predict(X_val) == Y_val).mean()
                    train_acc_history.append(train_acc)
                    val_acc_history.append(val_acc)
                    if best_val_acc < val_acc:
                        best_val_acc = val_acc
                        self.best_params = copy.deepcopy(self.params)

                if verbose and it % 100 == 0:
                    process_bar.set_description('iteration: {} / ({}), loss: {:.6f}, train acc: {:.2f}, val acc: {:.2f}' .format(it, num_iters, loss_batch, train_acc, val_acc))
                    
        # SGD vertion-2:    
        else:
            
            for epoch in range(num_epoch):
                if verbose:
                    print('='*70)
                    print('Training epoch {}/({})' .format(epoch+1, self.num_epoch))
                    print('-'*70)
                #########################################################################
                # TODO: Create a random minibatch of training data and labels, storing  #
                # them in X_batch and y_batch respectively.                             #
                #########################################################################
                idx = np.random.permutation(num_train_data)
                num_iteration = int(np.ceil(float(num_train_data)/self.batch_size)) 
                
                epoch_train_loss = 0# loss accumulation
                
                if verbose:
                    process_bar = tqdm(range(num_iteration))
                else:
                    process_bar = range(num_iteration)
                for it in process_bar:# iteration over each minibatch

                    start_idx = (it*self.batch_size)%num_train_data
                    X_batch = X[idx[start_idx:start_idx+self.batch_size], :]
                    Y_batch = Y[idx[start_idx:start_idx+self.batch_size]]
                     
                    # Forword pass
                    prob_scores = self.forword(X_batch)

                    # Loss
                    loss_batch = self.loss(Y_batch, prob_scores)
                    loss_history_batch.append(loss_batch)
                    epoch_train_loss += loss_batch

                    # Calculate gradients
                    grads_batch = self.backword(X_batch, Y_batch, prob_scores)

                    # Update the parameters
                    self.optimizer(grads_batch,update_rule)
                    if verbose:
                        process_bar.set_description('iteration: {} / ({}), loss: {:.6f}' .format(it, num_iteration, loss_batch))
                    
                epoch_train_loss /= num_iteration    
                loss_history_epoch.append(epoch_train_loss)
                # Check accuracy
                train_acc = 100*(self.predict(X) == Y).mean()
                val_acc = 100*(self.predict(X_val) == Y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                if best_val_acc < val_acc:
                    best_val_acc = val_acc
                    self.best_params = copy.deepcopy(self.params)
                    
                if verbose:
                    print('epoch: {} / ({}), loss: {:.6f}, train acc: {:.2f}, val acc: {:.2f}' .format(epoch+1, self.num_epoch, epoch_train_loss, train_acc, val_acc))
            
        return {
            'loss_history_batch': loss_history_batch,
            'loss_history_epoch': loss_history_epoch,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }
    
    def predict(self, X, best_param=False):
        
        """
        Use the trained network to predict labels for data points. For each data 
        point we predict scores for each of the C classes, and assign each data 
        point to the class with the highest score. Here we will use only score not the probability socre

        Inputs:
            - X(N, D): Test data, X[i, :] is a i-th test sample want to classify.
            - best_param (Boolean): If true, then will use the best network's weights, else use the current
            network's weights.

        Returns:
            - Y_pred (N): Test data predicted labels, Y_pred[i] = c means that X[i] is predicted 
            to have class c, where 0 <= c < C.
        """
        
        Y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        if best_param:
            Z_1 = np.dot(X, self.best_params['W1']) + self.best_params['b1']
            O_1 = sigmoid_func(Z_1)
            scores = np.dot(O_1, self.best_params['W2']) + self.best_params['b2']   
        else:
            Z_1 = np.dot(X, self.params['W1']) + self.params['b1']
            O_1 = sigmoid_func(Z_1)
            scores = np.dot(O_1, self.params['W2']) + self.params['b2']
             
        Y_pred = np.argmax(scores, axis=1)
        ###########################################################################
        #                              END OF YOUR CODE                           #
        ###########################################################################

        return Y_pred
