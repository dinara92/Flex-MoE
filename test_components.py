"""
Component Test Script - Test each part of the pipeline independently

Usage: python test_components.py
"""

import sys
import torch
import numpy as np

print("="*80)
print("COMPONENT TEST - Optuna Integration")
print("="*80)
print()

# Test 1: Imports
print("[1/5] Testing imports...")
try:
    from models import FlexMoE
    from data import load_and_preprocess_adni_custom, create_loaders
    from utils import seed_everything, setup_logger
    import optuna
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Basic configuration
print("\n[2/5] Testing configuration...")
try:
    class Args:
        device = 0
        data = 'adni_custom'
        modality = 'AMD'
        preprocessed = True
        initial_filling = 'mean'
        num_workers = 1  # Reduced for testing
        pin_memory = True
        use_common_ids = False
        batch_size = 8
        num_patches = 16
        hidden_dim = 128
        n_full_modalities = 3
        
        # Model params
        num_experts = 8
        top_k = 2
        dropout = 0.3
        num_heads = 4
        num_routers = 1
        num_layers_enc = 1
        num_layers_fus = 1
        num_layers_pred = 1
        lr = 0.0001
        gate_loss_weight = 0.01
        warm_up_epochs = 0
    
    args = Args()
    seed_everything(0)
    print("✓ Configuration created")
except Exception as e:
    print(f"✗ Configuration failed: {e}")
    sys.exit(1)

# Test 3: Data loading
print("\n[3/5] Testing data loading...")
try:
    modality_dict = {'amyloid': 0, 'mri': 1, 'demographic': 2}
    
    data_dict, encoder_dict, labels, train_ids, valid_ids, test_ids, n_labels, \
    input_dims, transforms, masks, observed_idx_arr, full_modality_index = \
        load_and_preprocess_adni_custom(args, modality_dict)
    
    print(f"✓ Data loaded successfully")
    print(f"  - Train samples: {len(train_ids)}")
    print(f"  - Val samples: {len(valid_ids)}")
    print(f"  - Test samples: {len(test_ids)}")
    print(f"  - Number of classes: {n_labels}")
    print(f"  - Modalities: {list(encoder_dict.keys())}")
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: DataLoader creation
print("\n[4/5] Testing data loaders...")
try:
    train_loader, train_loader_shuffle, val_loader, test_loader = create_loaders(
        data_dict, observed_idx_arr, labels, train_ids, valid_ids, test_ids, 
        args.batch_size, args.num_workers, args.pin_memory, input_dims, 
        transforms, masks, args.preprocessed, args.use_common_ids
    )
    
    print(f"✓ Data loaders created")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")
except Exception as e:
    print(f"✗ DataLoader creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Model and forward pass
print("\n[5/5] Testing model creation and forward pass...")
try:
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"  - Using device: {device}")
    
    # Create model
    num_modalities = len(args.modality)
    fusion_model = FlexMoE(
        num_modalities, full_modality_index, args.num_patches, args.hidden_dim, 
        n_labels, args.num_layers_fus, args.num_layers_pred, args.num_experts, 
        args.num_routers, args.top_k, args.num_heads, args.dropout
    ).to(device)
    print("✓ Model created")
    
    # Count parameters
    total_params = sum(p.numel() for p in fusion_model.parameters())
    trainable_params = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\n  Testing forward pass...")
    for batch_idx, (batch_samples, batch_labels, batch_mcs, batch_observed) in enumerate(val_loader):
        batch_samples = {k: v.to(device) for k, v in batch_samples.items()}
        batch_labels = batch_labels.to(device)
        batch_mcs = batch_mcs.to(device)
        batch_observed = batch_observed.to(device)
        
        # Create missing embeddings
        missing_embeds = torch.randn(
            (2**num_modalities)-1, args.n_full_modalities, 
            args.num_patches, args.hidden_dim
        ).to(device)
        
        # Prepare fusion inputs
        fusion_input = []
        for modality, samples in batch_samples.items():
            mask = batch_observed[:, modality_dict[modality]]
            encoded_samples = torch.zeros((samples.shape[0], args.num_patches, args.hidden_dim)).to(device)
            if mask.sum() > 0:
                encoded_samples[mask] = encoder_dict[modality](samples[mask])
            if (~mask).sum() > 0:
                encoded_samples[~mask] = missing_embeds[batch_mcs[~mask], modality_dict[modality]]
            fusion_input.append(encoded_samples)
        
        # Forward pass
        outputs = fusion_model(*fusion_input, expert_indices=batch_mcs)
        
        print(f"✓ Forward pass successful")
        print(f"  - Batch size: {batch_labels.shape[0]}")
        print(f"  - Output shape: {outputs.shape}")
        print(f"  - Output range: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")
        
        # Test backward pass
        print("\n  Testing backward pass...")
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, batch_labels)
        gate_loss = fusion_model.gate_loss()
        total_loss = loss + args.gate_loss_weight * gate_loss
        
        optimizer = torch.optim.Adam(fusion_model.parameters(), lr=args.lr)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        print(f"✓ Backward pass successful")
        print(f"  - Task loss: {loss.item():.4f}")
        print(f"  - Gate loss: {float(gate_loss):.4f}")
        print(f"  - Total loss: {total_loss.item():.4f}")
        
        break  # Only test one batch
        
except Exception as e:
    print(f"✗ Model/forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Optuna basic functionality
print("\n[6/6] Testing Optuna...")
try:
    def dummy_objective(trial):
        x = trial.suggest_float('x', -10, 10)
        return (x - 2) ** 2
    
    study = optuna.create_study(direction='minimize')
    study.optimize(dummy_objective, n_trials=3, show_progress_bar=False)
    
    print(f"✓ Optuna working")
    print(f"  - Best value: {study.best_value:.4f}")
    print(f"  - Best params: {study.best_params}")
except Exception as e:
    print(f"✗ Optuna failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final summary
print("\n" + "="*80)
print("✓ ALL COMPONENT TESTS PASSED!")
print("="*80)
print("\nSystem is ready for hyperparameter tuning.")
print("\nNext steps:")
print("  1. Quick test: ./quick_test.sh")
print("  2. Single seed: python optuna_tune_single_seed.py --seed 0 --device 0 --n_trials 3 --max_epochs 5")
print("  3. Full run: python optuna_tune_single_seed.py --seed 0 --device 0 --n_trials 200")
print()