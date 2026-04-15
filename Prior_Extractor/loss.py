import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        
        target = self._one_hot_encoder(target)
        
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice
        return loss / self.n_classes

class FocalTverskyLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.3, beta=0.7, gamma=1.0, eps=1e-6):
        super().__init__()
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.eps = float(eps)

    def _one_hot(self, target):
        return F.one_hot(target.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        target_oh = self._one_hot(target)

        probs = probs.reshape(probs.shape[0], probs.shape[1], -1)        # (B,C,N)
        target_oh = target_oh.reshape(target_oh.shape[0], target_oh.shape[1], -1)

        tp = (probs * target_oh).sum(dim=2)                              # (B,C)
        fp = (probs * (1.0 - target_oh)).sum(dim=2)
        fn = ((1.0 - probs) * target_oh).sum(dim=2)

        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        loss = (1.0 - tversky)
        if self.gamma != 1.0:
            loss = torch.pow(loss, self.gamma)
        return loss.mean()


class JointLoss(nn.Module):
    def __init__(self, num_classes=6, lambda_sdf=1.0, class_weights=None):
        super(JointLoss, self).__init__()
        self.lambda_sdf = lambda_sdf
        
        if class_weights is not None:
            if isinstance(class_weights, torch.Tensor):
                self.class_weights = class_weights.float()
            else:
                self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.ce = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.class_weights = None
            self.ce = nn.CrossEntropyLoss()
            
        self.seg_weight = 1.0
        self.ftv = FocalTverskyLoss(num_classes, alpha=0.3, beta=0.7, gamma=1.0)

        self.sdf_band = 0.0  # 先保持 0.0=关闭；后面想只算边界带再开
        self.sdf_loss_fn = nn.SmoothL1Loss(beta=0.1, reduction="none")

        self.use_uncertainty = True
        self.log_var_seg = nn.Parameter(torch.tensor(0.0))
        self.log_var_sdf = nn.Parameter(torch.tensor(0.0))
        self.log_var_cls = nn.Parameter(torch.tensor(0.0))
        
        self.cls_loss_fn = nn.CrossEntropyLoss()
          
    def forward(self, seg_pred, seg_target, sdf_pred, sdf_target, cls_pred=None, cls_target=None):

        loss_ce = self.ce(seg_pred, seg_target)
        loss_ftv = self.ftv(seg_pred, seg_target)
        seg_loss = loss_ce + self.seg_weight * loss_ftv

        # SDF loss
        sdf_map = self.sdf_loss_fn(sdf_pred, sdf_target)  # (B,1,H,W)

        if self.sdf_band > 0.0:
            band_mask = (sdf_target.abs() < self.sdf_band).float()
            denom = band_mask.sum().clamp_min(1.0)
            sdf_loss = (sdf_map * band_mask).sum() / denom
        else:
            sdf_loss = sdf_map.mean()

        # Classification loss
        if cls_pred is not None and cls_target is not None:
            cls_loss = self.cls_loss_fn(cls_pred, cls_target)
        else:
            cls_loss = torch.tensor(0.0, device=seg_pred.device)

        if self.use_uncertainty:
            w_seg = torch.exp(-self.log_var_seg)
            w_sdf = torch.exp(-self.log_var_sdf)
            w_cls = torch.exp(-self.log_var_cls)
            total_loss = (w_seg * seg_loss + self.log_var_seg) + \
                         (w_sdf * (self.lambda_sdf * sdf_loss) + self.log_var_sdf) + \
                         (w_cls * cls_loss + self.log_var_cls)
        else:
            total_loss = seg_loss + self.lambda_sdf * sdf_loss + cls_loss

        return total_loss, seg_loss, sdf_loss, cls_loss

if __name__ == "__main__":

    loss_fn = JointLoss(num_classes=6, lambda_sdf=0.5)
    
    B, C, H, W = 2, 6, 256, 256
    seg_pred = torch.randn(B, C, H, W)
    seg_target = torch.randint(0, C, (B, H, W))
    sdf_pred = torch.randn(B, 1, H, W)
    sdf_target = torch.randn(B, 1, H, W)
    
    cls_pred = torch.randn(B, 5)
    cls_target = torch.randint(0, 5, (B,))
    
    total, seg_l, sdf_l, cls_l = loss_fn(seg_pred, seg_target, sdf_pred, sdf_target, cls_pred, cls_target)
    
    print(f"Total Loss: {total.item()}")
    print(f"Seg Loss: {seg_l.item()}")
    print(f"SDF Loss: {sdf_l.item()}")
    print(f"Cls Loss: {cls_l.item()}")
    print("Loss function test passed!")
