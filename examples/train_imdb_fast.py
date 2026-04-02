import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import sys
import os
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from eidos_nn.utils.imdb_utils import IMDBDataset
from eidos_nn.optim.fractal_optimizer import FractalOptimizer, FractalScheduler
from eidos_nn.models.eidos_measurement_driven import ProbableCollapseLayer
from eidos_nn.utils.modular_phase_norm import ModularPhaseNorm
from eidos_nn.layers.eidos_transform import eidosTransform
from eidos_nn.utils.ablation_norms import IdentityNorm, RMSNorm1d
from eidos_nn.utils.logger import eidosLogger


class EidosProbableIMDB(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        num_paths=9,
        num_layers=2,
        matcher_mode="geometric",
        matcher_collapse="truncate",
        norm_type="modular",
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Probable Reasoning Layers -> The "Fast" variant reasoning block
        self.layers = nn.ModuleList([
            ProbableCollapseLayer(
                d_model,
                num_paths=num_paths,
                dropout=0.0,
                gumbel_tau=1.0,
                matcher_mode=matcher_mode,
                matcher_collapse=matcher_collapse,
            )
            for _ in range(num_layers)
        ])
        
        # Norm and Project
        if norm_type == "modular":
            self.norm = ModularPhaseNorm(d_model, base=7)
        elif norm_type == "rms":
            self.norm = RMSNorm1d(d_model)
        elif norm_type == "none":
            self.norm = IdentityNorm()
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")
        self.final_proj = eidosTransform(
            d_model,
            2,
            num_rotation_planes=2,
            matcher_mode=matcher_mode,
            matcher_collapse=matcher_collapse,
        )

    def forward(self, x):
        # x: [B, Seq_Len]
        x = self.embedding(x) # -> [B, Seq_Len, d_model]
        
        for layer in self.layers:
            x = layer(x) # ProbableCollapseLayer expects [B, Seq, Dim] and handles it natively!
            
        x = self.norm(x)
        x = x.mean(dim=1) # Global Average Pooling
        return self.final_proj(x)


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, epoch, logger=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
    for batch_idx, (batch_x, batch_y) in enumerate(pbar):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)        
        loss.backward()
        
        # Minimal tension simulation for FractalOptimizer
        tension = 0.0
        optimizer.adapt_frequencies(tension)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()
        
        acc = 100. * correct / total
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.2f}%'})
        
        if logger and batch_idx % 50 == 0:
            logger.log_progress(epoch, batch_idx, len(dataloader), loss.item(), acc)
        
    return total_loss / len(dataloader), 100. * correct / total

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc="Evaluating"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
            
    return total_loss / len(dataloader), 100. * correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64) # Increased batch size for speed since this is fast variant
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=2) 
    parser.add_argument('--matcher-mode', choices=['geometric', 'canonical'], default='geometric')
    parser.add_argument('--matcher-collapse', choices=['truncate', 'fold'], default='truncate')
    parser.add_argument('--norm-type', choices=['modular', 'rms', 'none'], default='modular')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading IMDB Data...")
    train_dataset = IMDBDataset(split='train')
    test_dataset = IMDBDataset(split='test')
    test_dataset.set_vocab(train_dataset.vocab, train_dataset.word2idx, train_dataset.idx2word)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Initializing EidosProbableIMDB (dim={args.d_model})...")
    model = EidosProbableIMDB(
        vocab_size=len(train_dataset.vocab),
        d_model=args.d_model,
        num_layers=args.num_layers,
        matcher_mode=args.matcher_mode,
        matcher_collapse=args.matcher_collapse,
        norm_type=args.norm_type,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    print(f"Norm Type: {args.norm_type}")
    
    # Structured JSON logger
    logger = eidosLogger(
        run_name=f"imdb_d{args.d_model}",
        config={
            "model_name": "EidosProbableIMDB",
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "epochs": args.epochs,
            "params": total_params,
            "matcher_mode": args.matcher_mode,
            "matcher_collapse": args.matcher_collapse,
            "norm_type": args.norm_type,
        },
        log_dir="logs"
    )
    
    batch_scale = args.batch_size / 32.0  # Scale relative to base batch of 32
    optimizer = FractalOptimizer(model.parameters(), base_lr=args.lr, batch_scale=batch_scale)
    scheduler = FractalScheduler(optimizer, warmup_batches=5)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        import time
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion, device, epoch, logger=logger)
        test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
        epoch_time = time.time() - epoch_start
        
        print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"Test:  Loss={test_loss:.4f}, Acc={test_acc:.2f}%")
        
        logger.log_epoch({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "test_loss": round(test_loss, 4),
            "test_acc": round(test_acc, 2)
        })
        logger.log_timing(epoch, {"epoch_seconds": round(epoch_time, 2)})
        
        if test_acc > best_acc:
            best_acc = test_acc
            
    logger.finalize({"best_test_acc": round(best_acc, 2), "d_model": args.d_model})
    print(f"\nFinal Best Accuracy for dim {args.d_model}: {best_acc:.2f}%")
    print(f"Logs saved to: {logger.json_path}")

if __name__ == "__main__":
    main()
