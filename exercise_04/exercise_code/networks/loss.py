
import os
import pickle
import numpy as np
from exercise_code.networks.linear_model import *

class Loss(object):
    def __init__(self):
        self.grad_history = []
        
    def forward(self,y_out, y_truth):
        return NotImplementedError
    
    def backward(self,y_out, y_truth, upstream_grad=1.):
        return NotImplementedError
    
    def __call__(self,y_out, y_truth):
        loss = self.forward(y_out, y_truth) 
        grad = self.backward(y_out, y_truth)
        return (loss, grad)
    
class L1(Loss):
        
    def forward(self,y_out, y_truth):
        """
        Performs the forward pass of the L1 loss function.

        :param y_out: [N, ] array predicted value of your model. 
               y_truth: [N, ] array ground truth value of your training set. 
        :return: [N, ] array of L1 loss for each sample of your training set. 
        """
        result = None
        #########################################################################
        # TODO:                                                                 #
        # Implement the forward pass and return the output of the L1 loss.      #
        #########################################################################
        
        result = np.abs(y_truth - y_out)
        
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

        return result
    
    def backward(self,y_out, y_truth):
        """
        Performs the backward pass of the L1 loss function.

        :param y_out: [N, ] array predicted value of your model. 
               y_truth: [N, ] array ground truth value of your training set. 
        :return: [N, ] array of L1 loss gradients w.r.t y_out for 
                  each sample of your training set. 
        """
        gradient = None
        ###########################################################################
        # TODO:                                                                   #
        # Implement the backward pass. Return the gradient wrt y_out              #
        # hint: you may use np.where here.                                        #
        ###########################################################################


        result = (-y_truth+y_out)/np.abs(y_truth-y_out)
        gradient = np.where(y_truth-y_out!=0, result, 0)


        self.grad_history.append(gradient)
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################        
        return gradient

class MSE(Loss):
  

    def forward(self,y_out, y_truth):
        """
        Performs the forward pass of the MSE loss function.

        :param y_out: [N, ] array predicted value of your model. 
                y_truth: [N, ] array ground truth value of your training set. 
        :return: [N, ] array of MSE loss for each sample of your training set. 
        """    
        result = None
        #########################################################################
        # TODO:                                                                 #
        # Implement the forward pass and return the output of the MSE loss.     #
        #########################################################################

        result = (y_truth - y_out)**2
        
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################
        
        return result
    
    def backward(self,y_out, y_truth):
        """
        Performs the backward pass of the MSE loss function.

        :param y_out: [N, ] array predicted value of your model. 
               y_truth: [N, ] array ground truth value of your training set. 
        :return: [N, ] array of MSE loss gradients w.r.t y_out for 
                  each sample of your training set. 
        """
        gradient = None
        ###########################################################################
        # TODO:                                                                   #
        # Implement the backward pass. Return the gradient wrt y_out              #
        ###########################################################################

        gradient = 2*(y_out-y_truth)

        self.grad_history.append(gradient)

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################   
        return gradient    

class BCE(Loss):
  

    def forward(self,y_out, y_truth):
        """
        Performs the forward pass of the binary cross entropy loss function.

        :param y_out: [N, ] array predicted value of your model. 
                y_truth: [N, ] array ground truth value of your training set. 
        :return: [N, ] array of binary cross entropy loss for each sample of your training set. 
        """    
        result = None
        #########################################################################
        # TODO:                                                                 #
        # Implement the forward pass and return the output of the BCE loss.     #
        #########################################################################

        
        result = -np.multiply(y_truth, np.log(y_out))-np.multiply((1-y_truth), np.log(1-y_out))

        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################
        
        return result
    
    def backward(self,y_out, y_truth):
        """
        Performs the backward pass of the loss function.

        :param y_out: [N, ] array predicted value of your model. 
               y_truth: [N, ] array ground truth value of your training set. 
        :return: [N, ] array of binary cross entropy loss gradients w.r.t y_out for 
                  each sample of your training set. 
        """
        gradient = None

        ###########################################################################
        # TODO:                                                                   #
        # Implement the backward pass. Return the gradient wrt y_out              #
        ###########################################################################


        gradient = -np.multiply(y_truth, 1/y_out)+np.multiply((1-y_truth),1/(1-y_out))
        
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################   
        return gradient    
