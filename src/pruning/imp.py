import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy

class LotteryTicketPruner:
    def __init__(self, model, pruning_rate_fc=0.2, pruning_rate_conv=0.1):
        self.model = model
        self.pruning_rate_fc = pruning_rate_fc
        self.pruning_rate_conv = pruning_rate_conv
        self.initial_state_dict = copy.deepcopy(model.state_dict())
        self.masks = self._init_masks()

    def _init_masks(self):
        masks = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                masks[name] = torch.ones_like(module.weight)
        return masks

    def prune_step(self):
        """
        Performs one step of magnitude pruning and updates the masks.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # To prune p% of REMAINING weights, we use the custom mask
                # However, PyTorch's prune.l1_unstructured 'amount' as float
                # prunes a percentage of the *total* weights if no mask is present,
                # or a percentage of *unpruned* weights if we handle it correctly.
                
                # A more robust way to do iterative pruning as per the paper:
                # 1. Get the current weights
                # 2. Find the threshold for the smallest p% of non-zero weights
                # 3. Update the mask
                
                weight = module.weight.data.abs()
                mask = self.masks[name]
                
                # Only consider weights that are currently not pruned
                active_weights = weight[mask > 0]
                if len(active_weights) == 0:
                    continue
                    
                if isinstance(module, nn.Linear):
                    amount = self.pruning_rate_fc
                else:
                    amount = self.pruning_rate_conv
                
                # Calculate number of weights to prune in this step
                n_prune = int(len(active_weights) * amount)
                if n_prune > 0:
                    threshold = torch.sort(active_weights)[0][n_prune]
                    new_mask = (weight > threshold).float()
                    self.masks[name] = mask * new_mask
                
        # Apply the updated masks to the model
        self.apply_masks()

    def reset_weights(self):
        """
        Resets the remaining weights to their initial values.
        """
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    # Apply the mask to the initial weights
                    initial_weight = self.initial_state_dict[f"{name}.weight"]
                    module.weight.copy_(initial_weight * self.masks[name])
                    if module.bias is not None:
                        module.bias.copy_(self.initial_state_dict[f"{name}.bias"])

    def get_sparsity(self):
        """
        Calculates the current sparsity of the model.
        """
        total_params = 0
        zero_params = 0
        for name, mask in self.masks.items():
            total_params += mask.nelement()
            zero_params += torch.sum(mask == 0).item()
        return zero_params / total_params if total_params > 0 else 0

    def apply_masks(self):
        """
        Ensures masks are applied to the weights (useful after loading or resetting).
        """
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    module.weight.mul_(self.masks[name])
