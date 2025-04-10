import numpy as np                  # for math
import pandas as pd                 # for data storage
import random                       # for weight and bias generation
import time                         # to record training time
import matplotlib.pyplot as plt     # visuals

def main():
    print("Hello from example-app!")

def init_variables():
    # independent variables (requires scaling)
    age = np.array([60, 63, 17, 35, 90, 10])
    exercise = np.array([8, 2, 7, 4, 1, 9])
    diet = np.array([7, 2, 7, 5, 1, 7])

    # min-max scaling for independent features
    x1 = (age - 0) / (101 - 0)
    x2 = (exercise - 0) / (11 - 0)
    x3 = (diet - 0) / (11 - 0)

    # dependent variable (does not need to be scaled)
    y = np.array([7, 3, 9, 6, 1, 10])

    # generate a random number uniformly between lower and upper bound
    lower_bound = .1
    upper_bound = .9

    # initiate random weights
    w1 = random.uniform(lower_bound, upper_bound)
    w2 = random.uniform(lower_bound, upper_bound)
    w3 = random.uniform(lower_bound, upper_bound)
    w4 = random.uniform(lower_bound, upper_bound)
    w5 = random.uniform(lower_bound, upper_bound)
    w6 = random.uniform(lower_bound, upper_bound)
    w7 = random.uniform(lower_bound, upper_bound)
    w8 = random.uniform(lower_bound, upper_bound)

    # initiate bias - can be set to zero (unlike weights)
    b1 = 0
    b2 = 0
    b3 = 0

    # initiate hidden neuron values to store for backward propagation
    h1 = 0
    h2 = 0

    # learning rate
    lr = 0.005

    # epochs
    iterations = 2000

    return(x1, x2, x3, y, w1, w2, w3, w4, w5, w6, w7, w8, b1, b2, b3, h1, h2, lr, iterations)


# Activation function : identity function
def activ_func(x):
    return x # identity

# Derivative of the identity activation function wrt sum of inputs and weights equation (L)
def d_activ_func(x):
    return 1

# Mean Suared Error (MSE)
def MSE(predictions, targets):
    n = len(targets)
    return ((1/n)*np.sum(((predictions-targets)**2)))
    # ALT METHOD: divide by 2 to make the derivitive simpler. return ((1/2n)*np.sum((predictions-targets)**2))
    # ALT METHOD: derivitive would be...       ((1/n)*(predictions - targets))

# Derivitve of Mean Suared Error wrt to the activation equation (Y_pred)
def d_MSE(predictions, targets):
    n = len(targets)
    return ((1/n)*2*(predictions - targets))

def back_propagation_layer_2(predictions, targets, Z, h1, h2, parameter):
  # calculate the gradient for each parameter by multipling each partial derivitve (chain rule)

  # (1)
  one = d_MSE(predictions, targets)

  # (2)
  two = d_activ_func(Z)

  # (3)
  if parameter == 'w7':
    three = h1
  elif parameter == 'w8':
    three = h2
  # bias is 1 (not = bias) because it is not multiplied by an input variable
  elif parameter == 'b3':
    three  = 1
  else:
    print('invalid parameter')

  # calculate gradient
  # returns a vector of gradients the size of input vector
  # gradient sizes may be able to distinguish feature importance (or at least find the feature that has the potential to improve the model the most)
  gradient =  one * two * three

  # divide by n again to get the average loss per sample
  # returns the mean gradient for the vector of gradients
  avg_gradient = np.mean(gradient)

  return avg_gradient

def back_propagation_layer_1(predictions, targets, Z, parameter, w7, w8, x1, x2, x3):
  # calculate the gradient for each parameter by multipling each partial derivitve (chain rule)

  # (1)
  one = d_MSE(predictions, targets)

  # (2)
  two = d_activ_func(Z)

  # (3)
  if parameter == 'w1':
    three = w7
  elif parameter == 'w2':
    three = w7
  elif parameter == 'w3':
    three = w7
  elif parameter == 'w4':
    three = w8
  elif parameter == 'w5':
    three = w8
  elif parameter == 'w6':
    three = w8
  elif parameter == 'b1':
    three = 1
  elif parameter == 'b2':
    three = 1
  else:
    print('invalid parameter')

  # (4)
  four = 1

  # (5)
  if parameter == 'w1':
    five = x1
  elif parameter == 'w2':
    five = x2
  elif parameter == 'w3':
    five = x3
  elif parameter == 'w4':
    five = x1
  elif parameter == 'w5':
    five = x2
  elif parameter == 'w6':
    five = x3
  elif parameter == 'b1':
    five = 1
  elif parameter == 'b2':
    five = 1
  else:
    print('invalid parameter')

  # calculate gradient
  # returns a vector of gradients the size of input vector
  # gradient sizes may be able to distinguish feature importance (or at least find the feature that has the potential to improve the model the most)
  gradient = one * two * three * four * five

  # divide by n again to get the average loss per sample
  # returns the mean gradient for the vector of gradients
  avg_gradient = np.mean(gradient)

  return avg_gradient


def hidden_layer_model(x1, x2, x3, w1, w2, w3, w4, w5, w6, w7, w8, b1, b2, b3):

    h1 = w1*x1 + w2*x2 + w3*x3 + b1
    h2 = w4*x1 + w5*x2 + w6*x3 + b2
    Z = w7*h1 + w8*h2 + b3
    Y_pred = activ_func(Z)
    return Y_pred, Z, h1, h2


def mlp_health_score(input_1, input_2, input_3):
    start_time = time.time()
    data_storage = pd.DataFrame(columns=['Z', 'y_pred', 'RMSE', 'w1_gradient', 'w2_gradient', 'w3_gradient', 'w4_gradient', 'w5_gradient', 'w6_gradient','w7_gradient', 'w8_gradient', 'b1_gradient','b2_gradient','b3_gradient', 'w1', 'w2', 'w3','w4', 'w5', 'w6','w7', 'w8', 'b1', 'b2', 'b3'])
    
    x1, x2, x3, y, w1, w2, w3, w4, w5, w6, w7, w8, b1, b2, b3, h1, h2, lr, iterations = init_variables()
    for i in range(iterations):


        # Forward propagation
        Y_pred, Z, h1, h2 = hidden_layer_model(x1, x2, x3, w1, w2, w3, w4, w5, w6, w7, w8, b1, b2, b3)

        # calculate prediction error
        loss = MSE(Y_pred, y)
        RMSE = np.sqrt(loss)

        # Backward propagation
        # calculate the gradient for each parameter
        w1_gradient = back_propagation_layer_1(Y_pred, y, Z, 'w1', w7, w8, x1, x2, x3)
        w2_gradient = back_propagation_layer_1(Y_pred, y, Z, 'w2', w7, w8, x1, x2, x3)
        w3_gradient = back_propagation_layer_1(Y_pred, y, Z, 'w3', w7, w8, x1, x2, x3)
        w4_gradient = back_propagation_layer_1(Y_pred, y, Z, 'w4', w7, w8, x1, x2, x3)
        w5_gradient = back_propagation_layer_1(Y_pred, y, Z, 'w5', w7, w8, x1, x2, x3)
        w6_gradient = back_propagation_layer_1(Y_pred, y, Z, 'w6', w7, w8, x1, x2, x3)
        w7_gradient = back_propagation_layer_2(Y_pred, y, Z, h1, h2, 'w7')
        w8_gradient = back_propagation_layer_2(Y_pred, y, Z, h1, h2, 'w8')

        b1_gradient = back_propagation_layer_1(Y_pred, y, Z, 'b1', w7, w8, x1, x2, x3)
        b2_gradient = back_propagation_layer_1(Y_pred, y, Z, 'b2', w7, w8, x1, x2, x3)
        b3_gradient = back_propagation_layer_2(Y_pred, y, Z, h1, h2, 'b3')

        # update weights and bias
        # the gradient vector points in the direction that will reduce error the most.
        # this value is negitive so you must subtract the gradient from the weight so the negitives cancel.
        w1 -= lr * w1_gradient
        w2 -= lr * w2_gradient
        w3 -= lr * w3_gradient
        w4 -= lr * w4_gradient
        w5 -= lr * w5_gradient
        w6 -= lr * w6_gradient
        w7 -= lr * w7_gradient
        w8 -= lr * w8_gradient

        b1 -= lr * b1_gradient
        b2 -= lr * b2_gradient
        b3 -= lr * b3_gradient

        # store values in df
        data_storage.loc[i] = [Z, Y_pred, RMSE, w1_gradient, w2_gradient, w3_gradient, w4_gradient, w5_gradient, w6_gradient, w7_gradient, w8_gradient, b1_gradient,b2_gradient, b3_gradient, w1, w2, w3,w4, w5, w6,w7, w8, b1, b2, b3]
    end_time = time.time()
    print('Training Time: ', round(end_time - start_time, 4), 'Seconds')


        # pull the trained weights and bias values from the last row of the dataframe
    fw1 = data_storage.iloc[-1]['w1']
    fw2 = data_storage.iloc[-1]['w2']
    fw3 = data_storage.iloc[-1]['w3']
    fw4 = data_storage.iloc[-1]['w4']
    fw5 = data_storage.iloc[-1]['w5']
    fw6 = data_storage.iloc[-1]['w6']
    fw7 = data_storage.iloc[-1]['w7']
    fw8 = data_storage.iloc[-1]['w8']
    fb1 = data_storage.iloc[-1]['b1']
    fb2 = data_storage.iloc[-1]['b2']
    fb3 = data_storage.iloc[-1]['b3']

    input_1 = (input_2 - 0) / (101 - 0)
    input_2 = (input_2 - 0) / (11 - 0)
    input_3 = (input_3 - 0) / (11 - 0)

    h1 = (fw1*input_1) + (fw2*input_2) + (fw3*input_3) + fb1
    h2 = (fw4*input_1) + (fw5*input_2) + (fw6*input_3) + fb2
    Z = (fw7*h1) + (fw8*h2 + fb3)

    Y_pred = activ_func(Z)

    return Y_pred