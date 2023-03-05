import torch
from torch import nn
from torch.nn import functional as F

from src.models.components.gather_layer import gather

class LongShortAlignmentLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, video_features, text_features):
        pass
    
    
class VICRegLoss(nn.Module):
    # https://github.com/vturrisi/solo-learn/blob/main/solo/losses/vicreg.py
    def __init__(
        self,
        sim_loss_weight: float = 25.0,
        var_loss_weight: float = 25.0,
        cov_loss_weight: float = 1.0,
        ) -> None:
        """_summary_
        Args:
            sim_loss_weight (float, optional): _description_. Defaults to 25.0.
            var_loss_weight (float, optional): _description_. Defaults to 25.0.
            cov_loss_weight (float, optional): _description_. Defaults to 1.0.
        """
        super().__init__()
        
        self.sim_loss_weight = sim_loss_weight
        self.var_loss_weight = var_loss_weight
        self.cov_loss_weight = cov_loss_weight

    
    def invariance_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Computes mse loss given batch of projected features z1 from view 1 and
        projected features z2 from view 2.
        Args:
            z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
            z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        Returns:
            torch.Tensor: invariance loss (mean squared error).
        """

        return F.mse_loss(z1, z2)


    def variance_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Computes variance loss given batch of projected features z1 from view 1 and
        projected features z2 from view 2.
        Args:
            z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
            z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        Returns:
            torch.Tensor: variance regularization loss.
        """

        eps = 1e-4
        std_z1 = torch.sqrt(z1.var(dim=0) + eps)
        std_z2 = torch.sqrt(z2.var(dim=0) + eps)
        std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
        return std_loss


    def covariance_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Computes covariance loss given batch of projected features z1 from view 1 and
        projected features z2 from view 2.
        Args:
            z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
            z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        Returns:
            torch.Tensor: covariance regularization loss.
        """

        N, D = z1.size()

        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)
        cov_z1 = (z1.T @ z1) / (N - 1)
        cov_z2 = (z2.T @ z2) / (N - 1)

        diag = torch.eye(D, device=z1.device)
        cov_loss = cov_z1[~diag.bool()].pow_(2).sum() / D + cov_z2[~diag.bool()].pow_(2).sum() / D
        return cov_loss


    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor
    ) -> torch.Tensor:
        """Computes VICReg's loss given batch of projected features z1 from view 1 and
        projected features z2 from view 2.
        Args:
            z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
            z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        Returns:
            torch.Tensor: VICReg loss.
        """

        sim_loss = self.invariance_loss(z1, z2)

        # vicreg's official code gathers the tensors here
        # https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
        z1, z2 = gather(z1), gather(z2)

        var_loss = self.variance_loss(z1, z2)
        cov_loss = self.covariance_loss(z1, z2)

        loss = self.sim_loss_weight * sim_loss + self.var_loss_weight * var_loss + self.cov_loss_weight * cov_loss
        
        return loss