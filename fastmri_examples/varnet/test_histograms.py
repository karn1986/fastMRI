import torch
import numpy as np
# torch.manual_seed(42)
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
def hist_loss(X, Y, bins: int=100):
   
    X = X.view(-1)
    Y = Y.view(-1)
    xmin, xmax = float(X.min()), float(X.max())
    ymin, ymax = float(Y.min()), float(Y.max())
    xstep = (xmax - xmin)/bins
    ystep = (ymax - ymin)/bins
    hist_loss = torch.zeros(1, requires_grad=True).to(X)
    zero = torch.zeros(1, requires_grad=True).to(X)
    one = torch.ones(1, requires_grad=True).to(X)
    xones = X/X
    
    
    hist_x = torch.empty(bins)
    hist_y = torch.empty(bins)
    vdiff_hist_gt = torch.empty(bins)
    n = len(X)
    for i in range(bins):
        # x is the empirical distribution of the X
        # y is the empirical distribution of the Y
        x = torch.sum(torch.where(((X>xmin+i*xstep) & (X<xmin+(i+1)*xstep)), xones, zero))
        y = torch.sum(torch.where(((Y>ymin+i*ystep) & (Y<ymin+(i+1)*ystep)), one, zero))
        hist_x[i] = x
        hist_y[i] = y
        hist_loss = hist_loss + ((x-y)/n)**2

    hist_loss    = torch.sqrt(hist_loss)

    return hist_loss, hist_x, hist_y

x = torch.randn(10000, requires_grad=True)
y = torch.randn(10000)
z, hist_x, hist_y = hist_loss(x, y)
# z.backward()
# print(x.grad is None)

def differentiable_histogram(x, bins=100, min=0.0, max=1.0):

    hist_torch = torch.zeros(bins).to(x.device)
    zero = torch.zeros(1).to(x)
    delta = (max - min) / bins

    # BIN_Table = torch.arange(start=min-0.5*delta, end=max+delta, step=delta)

    for i in range(bins):
        # h_r = BIN_Table[dim].item()             # h_r
        # h_r_sub_1 = BIN_Table[dim - 1].item()   # h_(r-1)
        # h_r_plus_1 = BIN_Table[dim + 1].item()  # h_(r+1)

        # mask_sub = ((h_r > x) & (x >= h_r_sub_1)).float()
        # mask_plus = ((h_r_plus_1 > x) & (x >= h_r)).float()

        # hist_torch[dim-1] += torch.sum(((x - h_r_sub_1) * mask_sub))
        # hist_torch[dim-1] += torch.sum(((h_r_plus_1 - x) * mask_plus))
        h_l = min + i * delta
        h_u = min + (i + 1) * delta

        mask = ((h_u > x) & (x >= h_l))

        hist_torch[i] += torch.sum(torch.where(mask, x - h_l, zero))
        hist_torch[i] += torch.sum(torch.where(mask, h_u - x, zero))

    return hist_torch / delta

hist_x1 = differentiable_histogram(x, bins = 100, min = float(x.min()), max = float(x.max()))
hist_y1 = differentiable_histogram(y, bins = 100, min = float(y.min()), max = float(y.max()))
z2 = (hist_x1-hist_y1)/len(x)
z2 = torch.norm(z2)

hist_x2 = torch.histc(x, bins = 100, min = float(x.min()), max = float(x.max()))
hist_y2 = torch.histc(y, bins = 100, min = float(y.min()), max = float(y.max()))
z3 = (hist_x2-hist_y2)/len(x)
z3 = torch.norm(z3)

_, ax1 = plt.subplots()

kwargs = dict(alpha=0.3, step = 'mid')

temp = np.linspace(x.min().detach().numpy(), x.max().detach().numpy(), 100, endpoint=False)
ax1.fill_between(temp, hist_x.detach().numpy() ,label = 'My Version', **kwargs)
ax1.fill_between(temp, hist_x1.detach().numpy() ,label = 'Diff hist version', **kwargs)
ax1.fill_between(temp, hist_x2.detach().numpy() ,label = 'torch.histc version', **kwargs)
ax1.legend()

