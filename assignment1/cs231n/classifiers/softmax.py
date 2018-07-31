import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N, C = X.shape[0], W.shape[1]
  for i in range(N):
      f = np.dot(X[i], W)
      f -= np.max(f) # f.shape = C
      loss = loss + np.log(np.sum(np.exp(f))) - f[y[i]]
      dW[:, y[i]] -= X[i]
      s = np.exp(f).sum()
      for j in range(C):
          dW[:, j] += np.exp(f[j]) / s * X[i]
  loss = loss / N + 0.5 * reg * np.sum(W * W)
  dW = dW / N + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  C=W.shape[1]
  N=X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  score=X.dot(W)
  score=np.power(np.exp(1),score-score.max())
  score=score/score.sum(axis=1).reshape((N,1)).dot(np.ones((1,C)))
  score=-np.log(score)[np.arange(N),y]
  loss=score.sum()/N+reg*np.sum(W*W)

  S=X.dot(W)
  counts= np.exp(S)/np.exp(S).sum(axis=1).reshape(N,1)
  counts[range(N), y] -= 1
  dW = np.dot(X.T, counts)

  loss = loss / N + 0.5 * reg * np.sum(W * W)
  dW = dW / N + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

