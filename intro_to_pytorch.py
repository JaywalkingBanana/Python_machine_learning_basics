####Code copied from lecture slides on Computer Vision by Justin Johnson

import torch


######################################
basic version
######################################
device = torch.device('cpu')
N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in, device = device)
y = torch.randn(N, D_out, device = device)
w1 = torch.randn(D_in, H, device = device)
w2 = torch.randn(H, D_out, device = device)

learning_rate = 1e-6
for t in range(500):
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)
    loss = (y_pred - y).pow(2).sum()
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)
    
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
    print(w1)

######################################
1st improvement
######################################

device = torch.device('cpu')
N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
w1 = torch.randn(D_in, H, requires_grad=True)
w2 = torch.randn(H, D_out, requires_grad=True)

learning_rate = 1e-6
for t in range(500):

    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred - y).pow(2).sum()
    loss.backward()

    if t%50 == 0 :
        print(t, loss.item())
        
    
    with torch.no_grad():        
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        w1.grad.zero_()
        w2.grad.zero_()    
        
######################################
instability
######################################        
def sigmoid(x):
    return 1.0 / (1.0 + (-x).exp())
    

device = torch.device('cpu')
N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
w1 = torch.randn(D_in, H, requires_grad=True)
w2 = torch.randn(H, D_out, requires_grad=True)

learning_rate = 1e-6
for t in range(500):

    y_pred = sigmoid(x.mm(w1)).mm(w2)
    loss = (y_pred - y).pow(2).sum()
    
    loss.backward()
    if t%50 == 0 :
        print(t, loss.item())
        
    with torch.no_grad():        
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        w1.grad.zero_()
        w2.grad.zero_()

######################################
improvement
######################################
class Sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = 1.0 / (1.0 + (-x).exp())
        ctx.save_for_backward(y)
        return y
    
    @staticmethod
    def backward(ctx, grad_y):
        y, = ctx.saved_tensors
        grad_x = grad_y * y * (1.0 - y)
        return grad_x
    
def sigmoid(x):
    return Sigmoid.apply(x)
    
device = torch.device('cpu')
N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
w1 = torch.randn(D_in, H, requires_grad=True)
w2 = torch.randn(H, D_out, requires_grad=True)

learning_rate = 1e-6
for t in range(5000):

    y_pred = sigmoid(x.mm(w1)).mm(w2)
    loss = (y_pred - y).pow(2).sum()
    
    loss.backward()
    if t%50 == 0 :
        print(t, loss.item())
        
    with torch.no_grad():        
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        w1.grad.zero_()
        w2.grad.zero_()    

######################################
2nd improvement
######################################
device = torch.device('cpu')
N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)

learning_rate = 1e-6
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
for t in range(5000):

    y_pred = model(x)
    loss = torch.nn.functional.mse_loss(y_pred, y)
    
    loss.backward()
    if t%50 == 0 :
        print(t, loss.item())
        
    optimizer.step()
    optimizer.zero_grad()        
