import os
import torch
import numpy as np
import argparse
import json
from pathlib import Path
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
    parser = argparse.ArgumentParser(description='FlexMoE Final Training with Best Hyperparameters')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--data', type=str, default='adni_custom')
    parser.add_argument('--modality', type=str, default='AMD')
    parser.add_argument('--preprocessed', type=str2bool, default=True)
    parser.add_argument('--initial_filling', type=str, default='mean')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=str2bool, default=True)
    parser.add_argument('--use_common_ids', type=str2bool, default=False)
    parser.add_argument('--max_epochs', type=int, default=150)
    parser.add_argument('--save_models', type=str2bool, default=True)
    
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


def combine_train_val_ids(train_ids, val_ids):
    """Combine train and validation IDs"""
    return np.concatenate([train_ids, val_ids])


def train_final_model(seed_idx, args, best_params):
    """
    Train final model on train+val using best hyperparameters,
    then evaluate on test set.
    """
    print(f"\n{'='*80}")
    print(f"Training final model for seed {seed_idx}")
    print(f"{'='*80}\n")
    
    # Set hyperparameters from best_params
    args.lr = best_params['learning_rate']
    args.num_experts = best_params['num_experts']
    args.hidden_dim = best_params['hidden_dim']
    args.top_k = best_params['top_k']
    args.dropout = best_params['dropout']
    args.gate_loss_weight = best_params['gate_loss_weight']
    args.batch_size = best_params['batch_size']
    args.warm_up_epochs = best_params['warm_up_epochs']
    args.num_heads = best_params['num_heads']
    args.num_patches = best_params['num_patches']
    
    # Fixed architecture params
    args.num_routers = 1
    args.num_layers_enc = 1
    args.num_layers_fus = 1
    args.num_layers_pred = 1
    
    # Set seed
    seed_everything(seed_idx)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    num_modalities = len(args.modality)
    
    # Load data
    modality_dict = {'amyloid': 0, 'mri': 1, 'demographic': 2}
    args.n_full_modalities = len(modality_dict)
    
    data_dict, encoder_dict, labels, train_ids, valid_ids, test_ids, n_labels, input_dims, transforms, masks, observed_idx_arr, full_modality_index = load_and_preprocess_adni_custom(args, modality_dict)
    
    # Combine train and validation IDs
    combined_train_ids = combine_train_val_ids(train_ids, valid_ids)
    
    # Create loaders (using combined train+val for training, test for evaluation)
    # For training: use combined_train_ids as train, empty array as val, test_ids as test
    train_loader, train_loader_shuffle, _, test_loader = create_loaders(
        data_dict, observed_idx_arr, labels, combined_train_ids, np.array([]), test_ids,
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
    
    # Training loop on combined train+val
    print(f"Training on {len(combined_train_ids)} samples (train+val combined)")
    print(f"Test set: {len(test_ids)} samples\n")
    
    for epoch in trange(args.max_epochs, desc=f"Seed {seed_idx}"):
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
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.max_epochs} - Task Loss: {np.mean(task_losses):.4f}, Gate Loss: {np.mean(gate_losses):.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    fusion_model.eval()
    for encoder in encoder_dict.values():
        encoder.eval()
    
    with torch.no_grad():
        test_preds, test_labels, test_probs = run_epoch(
            args, test_loader, encoder_dict, modality_dict, 
            missing_embeds, fusion_model, criterion, device
        )
    
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='macro')
    test_auc = roc_auc_score(test_labels, test_probs, multi_class='ovr')
    
    print(f"\n{'='*80}")
    print(f"Seed {seed_idx} Test Results:")
    print(f"  Accuracy: {test_acc*100:.2f}%")
    print(f"  F1 Score: {test_f1*100:.2f}%")
    print(f"  AUC:      {test_auc*100:.2f}%")
    print(f"{'='*80}\n")
    
    # Save model if requested
    if args.save_models:
        save_dir = Path('final_models')
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / f'final_model_seed_{seed_idx}.pth'
        
        torch.save({
            'seed': seed_idx,
            'hyperparameters': best_params,
            'missing_embeds': missing_embeds,
            'fusion_model': fusion_model.state_dict(),
            'encoder_dict': {modality: encoder.state_dict() for modality, encoder in encoder_dict.items()},
            'test_acc': test_acc,
            'test_f1': test_f1,
            'test_auc': test_auc
        }, save_path)
        
        print(f"Model saved to {save_path}")
    
    return test_acc, test_f1, test_auc


def main():
    args = parse_args()
    
    # Check if optuna results exist
    optuna_dir = Path('optuna_results')
    if not optuna_dir.exists():
        print("ERROR: optuna_results directory not found!")
        print("Please run optuna_tune.py first to generate hyperparameters.")
        return
    
    # Train final models for all 5 seeds
    test_accs = []
    test_f1s = []
    test_aucs = []
    
    for seed_idx in range(5):
        # Load best hyperparameters for this seed
        params_file = optuna_dir / f'best_params_seed_{seed_idx}.json'
        
        if not params_file.exists():
            print(f"WARNING: {params_file} not found. Skipping seed {seed_idx}")
            continue
        
        with open(params_file, 'r') as f:
            best_params = json.load(f)
        
        print(f"\nLoaded best hyperparameters for seed {seed_idx}:")
        print(f"  Validation F1: {best_params['best_val_f1']:.4f}")
        
        # Train final model
        test_acc, test_f1, test_auc = train_final_model(seed_idx, args, best_params)
        
        test_accs.append(test_acc)
        test_f1s.append(test_f1)
        test_aucs.append(test_auc)
    
    # Aggregate and report final results
    print(f"\n{'='*80}")
    print("FINAL TEST RESULTS ACROSS ALL SEEDS")
    print(f"{'='*80}\n")
    
    print(f"Accuracy:  {np.mean(test_accs)*100:.2f} ± {np.std(test_accs)*100:.2f}")
    print(f"F1 Score:  {np.mean(test_f1s)*100:.2f} ± {np.std(test_f1s)*100:.2f}")
    print(f"AUC:       {np.mean(test_aucs)*100:.2f} ± {np.std(test_aucs)*100:.2f}")
    
    print(f"\nPer-seed results:")
    for i, (acc, f1, auc) in enumerate(zip(test_accs, test_f1s, test_aucs)):
        print(f"  Seed {i}: Acc={acc*100:.2f}%, F1={f1*100:.2f}%, AUC={auc*100:.2f}%")
    
    # Save final results
    final_results = {
        'test_accuracy': {
            'mean': float(np.mean(test_accs)),
            'std': float(np.std(test_accs)),
            'values': [float(x) for x in test_accs]
        },
        'test_f1': {
            'mean': float(np.mean(test_f1s)),
            'std': float(np.std(test_f1s)),
            'values': [float(x) for x in test_f1s]
        },
        'test_auc': {
            'mean': float(np.mean(test_aucs)),
            'std': float(np.std(test_aucs)),
            'values': [float(x) for x in test_aucs]
        }
    }
    
    with open('final_test_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("Final results saved to 'final_test_results.json'")
    print(f"{'='*80}\n")
    
    # Create summary log
    logger = setup_logger('./logs', 'final_results', 'summary.txt')
    logger.info(f"{'='*80}")
    logger.info("FINAL TEST RESULTS - FlexMoE with Optuna Hyperparameter Tuning")
    logger.info(f"{'='*80}")
    logger.info(f"Accuracy:  {np.mean(test_accs)*100:.2f} ± {np.std(test_accs)*100:.2f}")
    logger.info(f"F1 Score:  {np.mean(test_f1s)*100:.2f} ± {np.std(test_f1s)*100:.2f}")
    logger.info(f"AUC:       {np.mean(test_aucs)*100:.2f} ± {np.std(test_aucs)*100:.2f}")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    main()