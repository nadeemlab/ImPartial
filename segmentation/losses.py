import torch
import torch.nn as nn
import sys
import torch.nn.functional as F

sys.path.append('../')

from general.losses import gradientLoss2d

class FocalLossBinary(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLossBinary, self).__init__()
        print("Focal Loss for BCE ...")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, outputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')

        # Compute the focal weight (alpha * (1 - pt)^gamma)
        pt = torch.exp(-bce_loss)
        if self.alpha is not None:
            focal_weight = self.alpha * (1 - pt)**self.gamma
        else:
            focal_weight = (1 - pt)**self.gamma

        # Combine the focal weight with the binary cross-entropy loss
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ImpartialLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.1, gamma=0.1):     # orig : alpha=0.9, beta=0.1, gamma=0.0
        super(ImpartialLoss, self).__init__()
        print("ImpartialLoss Loss ...")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.mse_loss = torch.nn.MSELoss()
        self.mumford_loss = gradientLoss2d()

    def forward(self, outputs, targets, inputs):
        loss_bce = self.bce_loss(outputs[:,0,:,:], targets[:,0,:,:])
        loss_mse = self.mse_loss(outputs[:,1:,:,:], inputs)
        loss_mumford = self.mumford_loss(outputs[:,0,:,:].unsqueeze(1))
        
        loss = loss_bce * self.alpha + loss_mse * self.beta + loss_mumford  * self.gamma

        return loss.mean()


if __name__ == '__main__':


    criterio = ImpartialLoss()
    inputs = torch.rand(5,2,256,256)
    targets = torch.rand(5,1,256,256)

    outputs = torch.rand(5,3,256,256)
    
    loss = criterio(outputs, targets, inputs)

    print("impartial loss = ", loss)




# # Example usage:
# # Assuming you have your model and data ready
# model = YourModel()
# criterion = FocalLossBinary(alpha=None, gamma=2, reduction='mean')
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # Inside your training loop
# outputs = model(inputs)
# loss = criterion(outputs, targets)
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
