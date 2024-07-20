import matplotlib.pyplot as plt
import numpy as np
from utils_log import sigmoid, map_feature

## Data plotting

def plot_data(X, y, pos_label="y=1", neg_label="y=0"):
    positive = y == 1
    negative = y == 0
    
    # Plot examples
    plt.plot(X[positive, 0], X[positive, 1], 'k+', label=pos_label) # x=depth, y=SPT-N
    plt.plot(X[negative, 0], X[negative, 1], 'yo', label=neg_label) # x=depth, y=SPT-N
    
    
def plot_decision_boundary(w, b, X, y, degree):
    # Credit to dibgerge on Github for this plotting code
     
    plot_data(X, y)
    
    if X.shape[1] <= 2:
        plot_x = np.array([min(X[:, 0]), max(X[:, 0])])
        plot_y = (-1. / w[1]) * (w[0] * plot_x + b)
        
        plt.plot(plot_x, plot_y, c="b")
        
    else:
        pad = 0.1 # padding for xlim, ylim
        xx, yy = np.meshgrid(np.linspace(min(X[:,0]) -pad, max(X[:,0]) +pad, 50), np.linspace(min(X[:,1]) -pad, max(X[:,1]) +pad, 50))  # x=depth, y=SPT-N
        x_in = np.c_[xx.ravel(), yy.ravel()]
        Z = sigmoid(np.dot(map_feature(x_in, degree), w) + b)

        Z = Z.reshape(xx.shape)
        
        # Plot z = 0.5
        plt.contour(xx,yy,Z, levels = [0.5], colors="g")