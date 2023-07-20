import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossBinary(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLossBinary, self).__init__()
        print("Focal Loss for BCE ...")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

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
