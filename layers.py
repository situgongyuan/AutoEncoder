import numpy as np
import multiprocessing as mtp
import time
from utils import *
"""
    this script implements some layers usually used in neural network.
"""

def get_index(img_height,img_width,filter_height,filter_width,pad,stride):
  """get index of row and column of every pixel in every img block,
     this function can be reused by conv_forward_naive and max_pool_naive.
  """
  assert (img_width + 2 * pad - filter_width) % stride == 0, 'width does not work'
  assert (img_height + 2 * pad - filter_height) % stride == 0, 'height does not work'
  out_height = (img_height + 2 * pad - filter_height) / stride + 1
  out_width = (img_width + 2 * pad - filter_width) / stride + 1

  #computer the row index,it's dimension is (out_height * out_weight,filter_height * filter_width)
  a1 = np.repeat(np.arange(filter_height),filter_width)
  b1 = np.tile(a1,out_width)
  c1 = b1 + np.arange(0,out_height * stride,stride).reshape(-1,1)
  row_index = c1.reshape(-1,filter_height * filter_width)

  #compute the column index,it's dimension is (out_height * out_weight,filter_height * filter_width)
  a_prime = np.repeat(np.arange(0,out_width * stride,stride),filter_height * filter_width)
  b_prime = np.tile(a_prime, out_height)
  c_prime = b_prime.reshape(-1,filter_width * filter_height)
  col_index = c_prime + np.tile(np.arange(filter_width),filter_height)

  return out_height,out_width,row_index,col_index

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)

  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  x_flat = x.reshape(x.shape[0],-1)
  out = np.dot(x_flat,w) + b
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx_flat = np.dot(dout,w.T)
  dx = dx_flat.reshape(x.shape)
  x_flat = x.reshape(x.shape[0],-1)
  dw = np.dot(x_flat.T,dout)
  db = np.sum(dout, axis = 0)

  return dw, db, dx


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = np.maximum(0,x)
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  dout[x <= 0] = 0
  dx = dout
  return dx

def sigmoid_forward(x):
    out = 1 / (1 + np.exp(-x))
    cache = out
    return out, cache

def sigmoid_backward(dout,cache):
    x = cache
    return dout * x * (1 - x)

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  N, C, H, W = x.shape
  num_filters, _, filter_height, filter_width = w.shape
  stride, pad = conv_param['stride'], conv_param['pad']

  x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
  out_height,out_width,row_index,col_index = get_index(H,W,filter_height,filter_width,pad,stride)

  x_col = x_padded[:,:,row_index,col_index]  #(10000,3,1024,9)
  x_col = x_col.transpose(0,2,1,3).reshape(N,out_width * out_height,-1)     #(10000,1024,27)

  w_prime = w.reshape(num_filters,-1).T   #(27,4)
  out = np.dot(x_col,w_prime) + b    #(10000,1024,4)
  out = out.transpose(0,2,1).reshape(N,num_filters,out_height,out_width)   #(10000,4,32,32)
  cache = (x_padded,w,b,row_index,col_index,conv_param,x_col)
  return out,cache

def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  x_padded,w,b,row_index,col_index,conv_param,x_col = cache
  N,C,x_padded_height,x_padded_width = x_padded.shape
  num_filters, _, filter_height, filter_width = w.shape
  _,_,out_height, out_width = dout.shape    #(10000,4,32,32)

  pad = conv_param['pad']

  trans_dout = dout.reshape(N,num_filters,-1).transpose(0,2,1)   #(10000,1024,4)
  dx_col = np.dot(trans_dout, w.reshape(num_filters,-1))  #(10000,1024,27)
  dx_col = dx_col.reshape(N,out_height * out_width,C,-1).transpose(0,2,1,3)   #(10000,3,1024,9)
  dx = np.zeros((N,C,x_padded_height,x_padded_width))
  np.add.at(dx,(slice(None),slice(None),row_index,col_index),dx_col)
  dx = dx[:,:,pad:-pad,pad:-pad]

  dout_reshape = trans_dout.reshape(-1,num_filters).T   #(4,10000 * 1024)
  x_col_reshape = x_col.reshape(-1,C * filter_height * filter_width)   #(10000 * 1024,27)
  dw = np.dot(dout_reshape,x_col_reshape).reshape(w.shape)

  db = np.sum(dout,axis = (0,2,3))  #(4,)

  return dx, dw, db

def max_pool_forward_reshape(x, pool_param):
  """
  A fast implementation of the forward pass for the max pooling layer that uses
  some clever reshaping.

  This can only be used for square pooling regions that tile the input.
  """
  N, C, H, W = x.shape
  pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
  stride = pool_param['stride']
  assert pool_height == pool_width == stride, 'Invalid pool params'
  assert H % pool_height == 0
  assert W % pool_height == 0
  x_reshaped = x.reshape(N, C, H / pool_height, pool_height,
                         W / pool_width, pool_width)
  out = x_reshaped.max(axis=5).max(axis=3)
  '''x_reshaped = x.reshape(N, C, H / pool_height, pool_height,
                         W / pool_width, pool_width).transpose(0,1,2,4,3,5)
  out = x_reshaped.max(axis = (4,5))'''

  cache = (x, x_reshaped, out)
  return out, cache

def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']
  same = (pool_height == pool_width == stride)
  if same:
    out, reshape_cache = max_pool_forward_reshape(x, pool_param)
    cache = ('reshape', reshape_cache)
  else:
    N,C,H,W = x.shape
    out_height,out_width,row_index,col_index = get_index(H,W,pool_height,pool_width,0,stride)
    x_col = x[:,:,row_index,col_index]
    i = np.argmax(x_col.reshape(-1,pool_width * pool_height),axis = 1)
    out_col = np.max(x_col,axis = 3)
    out = out_col.reshape(N,C,out_height,out_width)
    cache = ('im2col',x, pool_param,x_col,i,row_index,col_index)
  return out, cache

def max_pool_backward_reshape(dout, cache):
  """
  A fast implementation of the backward pass for the max pooling layer that
  uses some clever broadcasting and reshaping.

  This can only be used if the forward pass was computed using
  max_pool_forward_reshape.

  NOTE: If there are multiple argmaxes, this method will assign gradient to
  ALL argmax elements of the input rather than picking one. In this case the
  gradient will actually be incorrect. However this is unlikely to occur in
  practice, so it shouldn't matter much. One possible solution is to split the
  upstream gradient equally among all argmax elements; this should result in a
  valid subgradient. You can make this happen by uncommenting the line below;
  however this results in a significant performance penalty (about 40% slower)
  and is unlikely to matter in practice so we don't do it.
  """
  x, x_reshaped, out = cache

  x_reshaped = x_reshaped.transpose(0,1,2,4,3,5) #(100,3,16,16,2,2)
  out_newaxis = out[:, :, :, :, np.newaxis, np.newaxis]  #(100,3,16,16,1,1)
  mask = (x_reshaped == out_newaxis) #(100,3,16,16,2,2)
  dx_reshaped = mask * dout[:, :, :, :, np.newaxis,np.newaxis]
  dx = dx_reshaped.transpose(0,1,2,4,3,5).reshape(x.shape)
  return dx

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  if cache[0] == 'reshape':
    return max_pool_backward_reshape(dout, cache[1])
  else:
    _,x,pool_param,x_col,i,row_index,col_index = cache
    N,C,_,_ = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    _,_,out_height,out_width = dout.shape
    dx = np.zeros(x.shape)

    dout_reshape = dout.reshape(-1)
    dx_col = np.zeros(x_col.shape)   #(N,C,out_height * out_width, pool_height * pool_width)
    dx_col = dx_col.reshape(-1,pool_height * pool_width)

    dx_col[np.arange(N * C * out_width * out_height),i] = dout_reshape
    dx_col = dx_col.reshape(N,C,-1,pool_width * pool_height)

    np.add.at(dx,(slice(None),slice(None),row_index,col_index),dx_col)

  return dx


def cross_entropy_loss(x,y):
    n = x.shape[0]
    x_prime = sigmoid(x)
    loss = -1.0 * np.sum((y * np.log(x_prime) + (1 - y) * np.log(1 - x_prime))) / n
    dscore = (x_prime - y) / n
    return loss,dscore

def square_loss(x,y):
    n = x.shape[0]
    x_prime = sigmoid(x)
    loss = 0.5 * np.sum((x_prime - y)*(x_prime - y)) / n
    dscore = (x_prime - y) * x_prime * (1.0 - x_prime) / n
    return loss,dscore

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N

  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

if __name__ == "__main__":
  x_shape = (2, 3, 4, 4)
  w_shape = (3, 3, 4, 4)
  x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
  w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
  #b = np.linspace(-0.1, 0.2, num=3)
  b = np.zeros(3)
  conv_param = {'stride': 2, 'pad': 1}
  out,_ = conv_forward_naive(x,w,b,conv_param)
  print out