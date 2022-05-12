import numpy as np

#Generate artificial data
true_w = np.array([1, 2, 3, 4, 5])
d = true_w.shape[0]
points = []
for i in range(100000):
    x = np.random.randn(d)
    y = true_w.dot(x) + 0.1 * np.random.randn()
    points.append((x, y))

#define loss function
def sF(w, i):
    x, y = points[i]
    return (w.dot(x) - y) ** 2

#define gradient of loss
def sdF(w, i):
    x, y = points[i]
    return 2 * (w.dot(x) - y) * x 


def stochasticGradientDescent(sF, sdF, d, n):
    w = np.zeros(d)  #define weigths 
    eta = 1  #initial learning rate  
    nUpdates = 1
    
#run stochstic gradient descent for 1000 iterations    
    for t in range(1000):
        for i in range(n):
            value = sF(w, i)
            gradient = sdF(w, i)          
            eta = 1. / np.sqrt(nUpdates)  #update learning rate  
            nUpdates += 1         
            w = w - eta * gradient  #update weigths   
        print('iteration {}: w = {}, F(w) = {}'.format(t + 1, w, value)) 

stochasticGradientDescent(sF, sdF, d, len(points))        
