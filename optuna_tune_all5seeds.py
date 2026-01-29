import os
import torch
import numpy as np
import argparse
import json
from pathlib import Path
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from copy import deepcopy

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import trange
from models import FlexMoE
from utils import seed_everything, setup_logger
from data import load_and_preprocess_adni_custom, create_loaders
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="os.fork()")


def str2bool(s):
    if s not in {'False', 'True', 'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return (s == 'True') or (s == 'true')


def parse_args():
    parser = argparse.ArgumentParser(description='FlexMoE Optuna Hyperparameter Tuning')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--data', type=str, default='adni_custom')
    parser.add_argument('--modality', type=str, default='AMD')  # A=amyloid, M=mri, D=demographic
    parser.add_argument('--preprocessed', type=str2bool, default=True)
    parser.add_argument('--initial_filling', type=str, default='mean')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=str2bool, default=True)
    parser.add_argument('--use_common_ids', type=str2bool, default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_trials', type=int, default=200)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--early_stopping_patience', type=int, default=15)
    
    return parser.parse_args()


def run_epoch(args, loader, encoder_dict, modality_dict, missing_embeds, fusion_model, criterion, device, is_training=False, optimizer=None, gate_loss_weight=0.0):
    """Run one epoch of training or evaluation"""
    all_preds = []
    all_labels = []
    all_probs = []
    task_losses = []
    gate_losses = []
    
    if is_training:
        fusion_model.train()
        for encoder in encoder_dict.values():
            encoder.train()
    else:
        fusion_model.eval()
        for encoder in encoder_dict.values():
            encoder.eval()

    for batch_samples, batch_labels, batch_mcs, batch_observed in loader:
        batch_samples = {k: v.to(device, non_blocking=True) for k, v in batch_samples.items()}
        batch_labels = batch_labels.to(device, non_blocking=True)
        batch_mcs = batch_mcs.to(device, non_blocking=True)
        batch_observed = batch_observed.to(device, non_blocking=True)
        
        fusion_input = []
        num_modalities = len(args.modality)
        for i, (modality, samples) in enumerate(batch_samples.items()):
            mask = batch_observed[:, modality_dict[modality]]
            encoded_samples = torch.zeros((samples.shape[0], args.num_patches, args.hidden_dim)).to(device)
            if mask.sum() > 0:
                encoded_samples[mask] = encoder_dict[modality](samples[mask])
            if (~mask).sum() > 0:
                encoded_samples[~mask] = missing_embeds[batch_mcs[~mask], modality_dict[modality]]
            fusion_input.append(encoded_samples)

        outputs = fusion_model(*fusion_input, expert_indices=batch_mcs)

        if is_training:
            optimizer.zero_grad()
            task_loss = criterion(outputs, batch_labels)
            task_losses.append(task_loss.item())
            gate_loss = fusion_model.gate_loss()
            gate_losses.append(float(gate_loss))
            loss = task_loss + gate_loss_weight * gate_loss
            loss.backward()
            optimizer.step()
        else:
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_probs.extend(torch.nn.functional.softmax(outputs, dim=1).detach().cpu().numpy())

    if is_training:
        return task_losses, gate_losses
    else:
        return all_preds, all_labels, all_probs


def objective(trial, args, base_seed):
    """Optuna objective function for hyperparameter tuning"""
    
    # Suggest hyperparameters
    args.lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    args.num_experts = trial.suggest_categorical('num_experts', [4, 8, 16, 32])
    args.hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
    args.top_k = trial.suggest_categorical('top_k', [1, 2, 4])
    args.dropout = trial.suggest_float('dropout', 0.1, 0.5)
    args.gate_loss_weight = trial.suggest_float('gate_loss_weight', 1e-4, 1e-1, log=True)
    args.batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    args.warm_up_epochs = trial.suggest_int('warm_up_epochs', 0, 10)
    args.num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
    args.num_patches = trial.suggest_categorical('num_patches', [8, 16, 32])
    
    # Fixed architecture params
    args.num_routers = 1
    args.num_layers_enc = 1
    args.num_layers_fus = 1
    args.num_layers_pred = 1
    
    # Set seed for reproducibility
    seed_everything(base_seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    num_modalities = len(args.modality)
    
    # Load data
    modality_dict = {'amyloid': 0, 'mri': 1, 'demographic': 2}
    args.n_full_modalities = len(modality_dict)
    
    try:
        data_dict, encoder_dict, labels, train_ids, valid_ids, test_ids, n_labels, input_dims, transforms, masks, observed_idx_arr, full_modality_index = load_and_preprocess_adni_custom(args, modality_dict)
        
        train_loader, train_loader_shuffle, val_loader, test_loader = create_loaders(
            data_dict, observed_idx_arr, labels, train_ids, valid_ids, test_ids, 
            args.batch_size, args.num_workers, args.pin_memory, input_dims, 
            transforms, masks, args.preprocessed, args.use_common_ids
        )
        
        # Create models
        fusion_model = FlexMoE(
            num_modalities, full_modality_index, args.num_patches, args.hidden_dim, 
            n_labels, args.num_layers_fus, args.num_layers_pred, args.num_experts, 
            args.num_routers, args.top_k, args.num_heads, args.dropout
        ).to(device)
        
        # Setup parameters and optimizer
        params = list(fusion_model.parameters()) + [param for encoder in encoder_dict.values() for param in encoder.parameters()]
        
        if num_modalities > 1:
            missing_embeds = torch.nn.Parameter(
                torch.randn((2**num_modalities)-1, args.n_full_modalities, args.num_patches, args.hidden_dim, dtype=torch.float, device=device), 
                requires_grad=True
            )
            params += [missing_embeds]
        
        optimizer = torch.optim.Adam(params, lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop with early stopping
        best_val_f1 = 0.0
        patience_counter = 0
        
        for epoch in range(args.max_epochs):
            # Select appropriate loader
            if epoch >= args.warm_up_epochs:
                train_loader_current = train_loader_shuffle
            else:
                train_loader_current = train_loader
            
            # Training
            task_losses, gate_losses = run_epoch(
                args, train_loader_current, encoder_dict, modality_dict, 
                missing_embeds, fusion_model, criterion, device, 
                is_training=True, optimizer=optimizer, gate_loss_weight=args.gate_loss_weight
            )
            
            # Validation
            fusion_model.eval()
            for encoder in encoder_dict.values():
                encoder.eval()
            
            with torch.no_grad():
                val_preds, val_labels, val_probs = run_epoch(
                    args, val_loader, encoder_dict, modality_dict, 
                    missing_embeds, fusion_model, criterion, device
                )
            
            val_f1 = f1_score(val_labels, val_preds, average='macro')
            
            # Report intermediate value for pruning
            trial.report(val_f1, epoch)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return best_val_f1
    
    except Exception as e:
        print(f"Trial failed with error: {e}")
        raise optuna.TrialPruned()


def run_optuna_for_seed(seed_idx, args):
    """Run Optuna hyperparameter search for a specific seed"""
    print(f"\n{'='*80}")
    print(f"Running hyperparameter tuning for seed {seed_idx}")
    print(f"{'='*80}\n")
    
    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        study_name=f'flex_moe_seed_{seed_idx}',
        pruner=MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=5
        ),
        sampler=TPESampler(seed=seed_idx)
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, deepcopy(args), seed_idx),
        n_trials=args.n_trials,
        show_progress_bar=True
    )
    
    # Save results
    results_dir = Path('optuna_results')
    results_dir.mkdir(exist_ok=True)
    
    # Best hyperparameters
    best_params = study.best_params
    best_params['best_val_f1'] = study.best_value
    best_params['best_trial_number'] = study.best_trial.number
    
    with open(results_dir / f'best_params_seed_{seed_idx}.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # Save detailed study statistics
    study_stats = {
        'best_value': study.best_value,
        'best_trial': study.best_trial.number,
        'n_trials': len(study.trials),
        'best_params': best_params,
        'all_trials': [
            {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': str(trial.state)
            }
            for trial in study.trials
        ]
    }
    
    with open(results_dir / f'study_stats_seed_{seed_idx}.json', 'w') as f:
        json.dump(study_stats, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results for seed {seed_idx}")
    print(f"{'='*80}")
    print(f"Best F1 score: {study.best_value:.4f}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"\nBest hyperparameters:")
    for key, value in best_params.items():
        if key not in ['best_val_f1', 'best_trial_number']:
            print(f"  {key}: {value}")
    
    return best_params, study


def main():
    args = parse_args()
    
    # Run hyperparameter tuning for all 5 seeds
    all_best_params = []
    
    for seed_idx in range(5):  # Seeds 0-4
        best_params, study = run_optuna_for_seed(seed_idx, args)
        all_best_params.append(best_params)
    
    # Aggregate results across seeds
    print(f"\n{'='*80}")
    print("HYPERPARAMETER TUNING SUMMARY - ALL SEEDS")
    print(f"{'='*80}\n")
    
    val_f1_scores = [params['best_val_f1'] for params in all_best_params]
    print(f"Validation F1 across seeds:")
    print(f"  Mean: {np.mean(val_f1_scores):.4f}")
    print(f"  Std:  {np.std(val_f1_scores):.4f}")
    print(f"  Min:  {np.min(val_f1_scores):.4f}")
    print(f"  Max:  {np.max(val_f1_scores):.4f}")
    
    # Save aggregated results
    with open('optuna_results/all_seeds_summary.json', 'w') as f:
        json.dump({
            'mean_val_f1': float(np.mean(val_f1_scores)),
            'std_val_f1': float(np.std(val_f1_scores)),
            'min_val_f1': float(np.min(val_f1_scores)),
            'max_val_f1': float(np.max(val_f1_scores)),
            'all_best_params': all_best_params
        }, f, indent=2)
    
    print("\nHyperparameter tuning complete!")
    print("Results saved in 'optuna_results/' directory")
    print("\nNext step: Run final_training.py to train on train+val and evaluate on test")


if __name__ == '__main__':
    main()