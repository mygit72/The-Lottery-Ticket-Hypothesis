import torch

from models.lenet import LeNet300
from pruning.imp import LotteryTicketPruner


def make_pruner(pruning_rate_fc=0.5):
    model = LeNet300()
    return model, LotteryTicketPruner(model, pruning_rate_fc=pruning_rate_fc)


def test_initial_masks_are_all_ones():
    model, pruner = make_pruner()
    for name, module in model.named_modules():
        if name in pruner.masks:
            assert torch.equal(pruner.masks[name], torch.ones_like(module.weight))
    assert pruner.get_sparsity() == 0.0


def test_prune_step_removes_expected_fraction():
    model, pruner = make_pruner(pruning_rate_fc=0.5)

    pruner.prune_step()
    # ~50% of weights should now be pruned (allow slack for integer rounding per layer)
    assert abs(pruner.get_sparsity() - 0.5) < 0.02

    pruner.prune_step()
    # A second round prunes 50% of what's *left*, i.e. cumulative ~75%
    assert abs(pruner.get_sparsity() - 0.75) < 0.02


def test_prune_step_zeros_smallest_magnitude_weights():
    model, pruner = make_pruner(pruning_rate_fc=0.5)
    pruner.prune_step()

    for name, module in model.named_modules():
        if name in pruner.masks:
            mask = pruner.masks[name]
            pruned = mask == 0
            kept = mask == 1
            if pruned.any() and kept.any():
                # Every pruned weight's magnitude must be <= every kept weight's magnitude
                assert module.weight.data.abs()[pruned].max() <= module.weight.data.abs()[kept].min()


def test_reset_weights_restores_theta0_under_the_mask():
    model, pruner = make_pruner(pruning_rate_fc=0.5)
    initial_fc1_weight = pruner.initial_state_dict['fc1.weight'].clone()

    pruner.prune_step()
    # Perturb the weights, simulating a round of training
    with torch.no_grad():
        model.fc1.weight.add_(1.0)

    pruner.reset_weights()

    mask = pruner.masks['fc1']
    # Unpruned weights go back to theta_0; pruned weights are forced to zero
    assert torch.equal(model.fc1.weight.data, initial_fc1_weight * mask)


def test_apply_mask_to_model_matches_pruner_mask():
    model, pruner = make_pruner(pruning_rate_fc=0.3)
    pruner.prune_step()

    control_model = LeNet300()
    pruner.apply_mask_to_model(control_model)

    mask = pruner.masks['fc1']
    assert torch.equal((control_model.fc1.weight.data == 0), (mask == 0))
    # The control model keeps its own random init where the mask is 1 (not theta_0)
    assert not torch.equal(control_model.fc1.weight.data, model.fc1.weight.data)
