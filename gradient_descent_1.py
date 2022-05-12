
#create points
points = [(2, 4), (4, 2), (3, 3)]

#define loss function
def F(w):
    return sum((w * x - y) ** 2 for x, y in points)
#define gradient of loss
def dF(w):
    return sum(2 * (w * x - y) * x for x, y in points)

#define weight(slope)  
w = 0

#define learning rate
eta = 0.01

#run gradient descent for 100 epochs; print at each iteration the weight and loss
for t in range(100):
    value = F(w)
    gradient = dF(w)
    w = w - eta * gradient
    print('iteration {}: w = {}, F(w) = {}'.format(t + 1, w, value))
