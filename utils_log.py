import itertools
import numpy as np
import math

def load_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    X = data[:,[0,1,4]]   # data[:, [0,4]]
    y = data[:,6]
    return X, y


def initialize_weight(X):
    """
    Initialize weights and biases for the number of features
    """
    n = X.shape[1]
    
    w = np.zeros((1,n))
    b = 0.

    return w, b


def sigmoid(z):
    """
    Compute sigmoid of z
    """
    z = np.clip( z, -500, 500 ) # protect against overflow
    g = 1.0/(1.0+np.exp(-z))
    
    return g

# Data transformation
# 1. norm_zscore(X)
# 2. map_feature()

def norm_zscore(X):
    """
    Normalize data using z-score
    """
    m, n = X.shape
    X_norm = np.zeros((m,n))
    
    for j in range(n):
    
        # Compute mean and std.p
        mean = np.mean(X[:, j])
        std = np.std(X[:, j])
    
        for i in range(m):
            X_norm[i,j] = (X[i,j] - mean) / std
   
    return X_norm

def map_feature(X, degree):
    """
    Feature mapping function to polynomial features
    """
    
    X_mapped = np.hstack([[[np.prod(c) for c in itertools.combinations_with_replacement(row, i)] for row in X] for i in range(1, degree+1)])

    return X_mapped


# Regularization for logistic regression (to include lambda_)
# 1. compute_cost_logistic_reg()
# 2. compute_gradient_logistic_reg()

def compute_cost_logistic_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost
    """

    m, n  = X.shape
    cost = 0.
    
    for i in range(m):
        z_i = np.dot(X[i], w) + b                                      #(n,)(n,)=scalar, see np.dot
        f_wb_i = sigmoid(z_i)                                          #scalar
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)      #scalar
             
    cost = cost/m                                                      #scalar

    reg_cost = 0
    
    for j in range(n):
        reg_cost += (w[j]**2)                                          #scalar
        
    reg_cost = (lambda_/(2*m)) * reg_cost                              #scalar
    
    total_cost = cost + reg_cost                                       #scalar
    
    return total_cost 


def compute_gradient_logistic_reg(X, y, w, b, lambda_ = 1): 
    """
    Computes the gradient for logistic regression with regularization
    """
    m,n = X.shape
    dj_dw = np.zeros((n,))                            #(n,)
    dj_db = 0.0                                       #scalar

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          #(n,)(n,)=scalar
        err_i  = f_wb_i  - y[i]                       #scalar
        
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      #scalar
        dj_db = dj_db + err_i
        
    dj_dw = dj_dw/m                                   #(n,)
    dj_db = dj_db/m                                   #scalar

    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]

    return dj_db, dj_dw  

## Gradient descent

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking num_iters gradient steps with learning rate alpha
    """
    
    # number of training examples
    m = len(X)
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)   

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db              
       
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.3f}   ")
        
    return w_in, b_in, J_history, w_history #return w and J,w history for graphing



## Result prediction function

def predict(X, w, b): 
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters w

    """
    # number of training examples
    m, n = X.shape   
    p = np.zeros(m)
   
    # Loop over each example
    for i in range(m):   
        z_wb = 0
        # Loop over each feature
        for j in range(n): 
            # Add the corresponding term to z_wb
            z_wb += np.dot(w[j],X[i,j])
        
        # Add bias term 
        z_wb += b
        
        # Calculate the prediction for this example
        f_wb = sigmoid(z_wb)

        # Apply the threshold
        p[i] = f_wb >= 0.5
        
    return p 
