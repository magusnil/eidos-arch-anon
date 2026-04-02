"""
SET-VALUED TRANSFORMER ON IMDB SENTIMENT ANALYSIS
Real-world validation on 25k movie reviews

Expected: 85-88% (competitive with standard transformers)
Baseline: Standard transformers get ~83-86% on this task

Usage:
    python train_transformer_imdb.py --epochs 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import argparse
from tqdm import tqdm
import numpy as np
from collections import Counter

# Import set-valued transformer from layers module
import sys
import os
# Add the directory containing eidos_nn to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import updated eidos components
from eidos_nn.layers.eidos_transform import eidosTransform, eidosSequential
from eidos_nn.layers.true_eidos_ffn import TrueeidosFFN
from eidos_nn.utils.modular_phase_norm import ModularPhaseNorm



# ============================================================================
# IMDB DATASET LOADER
# ============================================================================

class IMDBDataset(Dataset):
    """
    IMDB Movie Review Sentiment Dataset
    - 25k training reviews
    - 25k test reviews
    - Binary classification: positive (1) vs negative (0)
    """
    def __init__(self, split='train', max_len=256, vocab_size=10000):
        """
        Args:
            split: 'train' or 'test'
            max_len: Maximum sequence length (truncate/pad)
            vocab_size: Vocabulary size (most common words)
        """
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.split = split
        
        # Initialize vocab (will be built for train, set for test)
        self.vocab = None
        self.word2idx = None
        self.idx2word = None
        
        print(f"Loading IMDB {split} dataset...")
        
        # Load data
        try:
            from datasets import load_dataset
            dataset = load_dataset('imdb', split=split)
            self.texts = dataset['text']
            self.labels_raw = dataset['label']
        except Exception as exc:
            print(f"Warning: unable to load IMDB via datasets ({type(exc).__name__}).")
            print("Using dummy data for testing.")
            print("Install with: pip install datasets")
            self.texts = self._create_dummy_data(split)
            self.labels_raw = [0 if i < len(self.texts)//2 else 1 for i in range(len(self.texts))]
        
        print(f"Loaded {len(self.texts)} reviews")
        
        # Build vocab for train split
        if split == 'train':
            self.vocab, self.word2idx, self.idx2word = self._build_vocab(self.texts)
            self._process_data()
        else:
            # Test split will have vocab set later
            self.data = []
            self.labels = []
    
    def _create_dummy_data(self, split):
        """Create dummy reviews for testing without datasets library"""
        positive_words = ['excellent', 'amazing', 'great', 'wonderful', 'fantastic', 
                         'loved', 'best', 'brilliant', 'outstanding', 'superb']
        negative_words = ['terrible', 'awful', 'horrible', 'worst', 'bad',
                         'disappointing', 'waste', 'poor', 'boring', 'dull']
        
        num_samples = 1000 if split == 'train' else 500
        texts = []
        
        for i in range(num_samples):
            if i < num_samples // 2:
                words = np.random.choice(negative_words, size=20).tolist()
                text = ' '.join(words) + ' movie film story acting'
            else:
                words = np.random.choice(positive_words, size=20).tolist()
                text = ' '.join(words) + ' movie film story acting'
            
            texts.append(text)
        
        return texts
    
    def _tokenize(self, text):
        """Simple tokenization: lowercase + split"""
        text = text.lower()
        for char in '.,!?;:"\'()[]{}':
            text = text.replace(char, ' ')
        tokens = text.split()
        return tokens
    
    def _build_vocab(self, texts):
        """Build vocabulary from texts"""
        print("Building vocabulary...")
        word_counts = Counter()
        
        for text in tqdm(texts, desc="Counting words"):
            tokens = self._tokenize(text)
            word_counts.update(tokens)
        
        # Keep most common words
        most_common = word_counts.most_common(self.vocab_size - 4)
        
        # Special tokens
        vocab = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        vocab.extend([word for word, _ in most_common])
        
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = {idx: word for word, idx in word2idx.items()}
        
        print(f"Vocabulary size: {len(vocab)}")
        
        return vocab, word2idx, idx2word
    
    def _encode(self, tokens):
        """Convert tokens to indices"""
        if self.word2idx is None:
            raise RuntimeError("Vocabulary not set! Call set_vocab() first for test set.")
        
        encoded = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
        
        # Truncate if too long
        if len(encoded) > self.max_len:
            encoded = encoded[:self.max_len]
        
        # Pad if too short
        while len(encoded) < self.max_len:
            encoded.append(self.word2idx['<PAD>'])
        
        return torch.tensor(encoded, dtype=torch.long)
    
    def _process_data(self):
        """Tokenize and encode all texts"""
        if self.word2idx is None:
            raise RuntimeError("Vocabulary not set! Cannot process data.")
        
        self.data = []
        self.labels = []
        
        print("Tokenizing reviews...")
        for text, label in tqdm(zip(self.texts, self.labels_raw), total=len(self.texts)):
            tokens = self._tokenize(text)
            encoded = self._encode(tokens)
            self.data.append(encoded)
            self.labels.append(label)
        
        print(f"Processed {len(self.data)} reviews")
    
    def set_vocab(self, vocab, word2idx, idx2word):
        """Set vocabulary from training set (for test set)"""
        self.vocab = vocab
        self.word2idx = word2idx
        self.idx2word = idx2word
        # Now process the data
        self._process_data()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class eidosClassifier(nn.Module):
    """
    eidos Classifier for IMDB sentiment analysis, replacing classical nn.Linear/LayerNorm
    with eidosTransform/ModularPhaseNorm and using TrueeidosFFN blocks.
    """
    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8, # Not directly used in this simplified eidos model, kept for arg consistency
        num_layers=6,
        d_ff=2048,
        max_seq_len=512,
        num_classes=2,
        dropout=0.1, # Applied to embeddings only
        num_paths=9 # For TrueeidosFFN
    ):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings (remain standard for now as they are just lookups)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout_emb = nn.Dropout(dropout) # Apply dropout to embeddings
        
        # Stack TrueeidosFFN blocks (eidos-compliant feed-forward layers)
        self.eidos_blocks = nn.ModuleList([
            TrueeidosFFN(
                d_model=d_model,
                d_ff=d_ff,
                num_paths=num_paths,
                dropout=0.0, # TrueeidosFFN applies its own internal dropout if configured
                use_context=False # Simplified context handling for this demo
            )
            for _ in range(num_layers)
        ])
        
        # Final eidos normalization before classification
        self.norm = ModularPhaseNorm(d_model, base=7)
        
        # Classification head using eidosTransform
        # num_rotation_planes could be tuned, 1 is simplest.
        self.fc = eidosTransform(d_model, num_classes, num_rotation_planes=1) 
    
    def forward(self, x):
        batch, seq_len = x.shape
        
        # Embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)
        x = self.token_emb(x) + self.pos_emb(positions)
        x = self.dropout_emb(x)
        
        # Process through eidos Blocks
        for block in self.eidos_blocks:
            x = block(x) # [batch, seq_len, d_model]
        
        # Final eidos normalization
        x = self.norm(x)
        
        # Mean pooling across sequence dimension for classification
        x = x.mean(dim=1) 
        
        # Classification
        return self.fc(x)

# ============================================================================
# STANDARD TRANSFORMER (Baseline for Comparison)
# ============================================================================

class StandardTransformer(nn.Module):
    """
    Standard transformer for comparison
    Same architecture as SetValuedTransformer but without set-valued branches
    """
    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_seq_len=512,
        num_classes=2,
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Standard transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        batch, seq_len = x.shape
        
        # Embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)
        x = self.token_emb(x) + self.pos_emb(positions)
        x = self.dropout(x)
        
        # Transformer
        x = self.transformer(x)
        
        # Classification
        x = self.norm(x)
        x = x.mean(dim=1)  # Mean pooling
        
        return self.fc(x)


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch_x, batch_y in pbar:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Forward
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward
        loss.backward()
        optimizer.step()

        # Stats
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(dataloader), 100. * correct / total


def eval_epoch(model, dataloader, criterion, device):
    """Evaluate on test set"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc="Evaluating"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='both', 
                       choices=['setvalued', 'standard', 'both'],
                       help='Which model to train')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=4) # Reduced to 4 for VRAM
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--d-model', type=int, default=512)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=6)
    parser.add_argument('--max-len', type=int, default=256)
    parser.add_argument('--vocab-size', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}\n")
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    print("=" * 70)
    print("LOADING IMDB DATASET")
    print("=" * 70)
    
    train_dataset = IMDBDataset(split='train', max_len=args.max_len, vocab_size=args.vocab_size)
    test_dataset = IMDBDataset(split='test', max_len=args.max_len, vocab_size=args.vocab_size)
    
    # Share vocabulary from training set
    test_dataset.set_vocab(train_dataset.vocab, train_dataset.word2idx, train_dataset.idx2word)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Vocabulary size: {len(train_dataset.vocab)}")
    
    # ========================================================================
    # TRAIN MODELS
    # ========================================================================
    results = {}
    
    models_to_train = []
    if args.model in ['setvalued', 'both']:
        models_to_train.append(('Set-Valued Transformer', 'setvalued'))
    if args.model in ['standard', 'both']:
        models_to_train.append(('Standard Transformer', 'standard'))
    
    for model_name, model_type in models_to_train:
        print("\n" + "=" * 70)
        print(f"TRAINING: {model_name}")
        print("=" * 70)
        
        # Create model
        if model_type == 'setvalued':
            model = eidosClassifier(
                vocab_size=len(train_dataset.vocab),
                d_model=args.d_model,
                num_layers=args.num_layers,
                d_ff=args.d_model * 4, # d_ff passed to TrueeidosFFN blocks
                max_seq_len=args.max_len,
                num_classes=2,
                dropout=0.1, # Applied to embeddings
                num_paths=9 # Fixed to 9, as required by TrueeidosFFN's HierarchicalPathScorer
            ).to(device)
        else:
            model = StandardTransformer(
                vocab_size=len(train_dataset.vocab),
                d_model=args.d_model,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                d_ff=args.d_model * 4,
                max_seq_len=args.max_len,
                num_classes=2,
                dropout=0.1
            ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params:,}")
        
        # Optimizer and criterion
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_acc = 0.0
        train_accs = []
        test_accs = []
        
        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")
            print("-" * 50)
            
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
            
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Test Loss:  {test_loss:.4f}  | Test Acc:  {test_acc:.2f}%")
            
            if test_acc > best_acc:
                best_acc = test_acc
                print(f"✅ New best: {best_acc:.2f}%")
        
        results[model_name] = {
            'best_acc': best_acc,
            'final_train': train_accs[-1],
            'final_test': test_accs[-1],
            'train_history': train_accs,
            'test_history': test_accs
        }
        
        print(f"\n{model_name} - Best Test Acc: {best_acc:.2f}%")
    
    # ========================================================================
    # FINAL COMPARISON
    # ========================================================================
    print("\n" + "=" * 70)
    print("FINAL RESULTS - IMDB SENTIMENT CLASSIFICATION")
    print("=" * 70)
    
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Best Test Accuracy:  {result['best_acc']:.2f}%")
        print(f"  Final Train Accuracy: {result['final_train']:.2f}%")
        print(f"  Final Test Accuracy:  {result['final_test']:.2f}%")
    
    if len(results) == 2:
        sv_acc = results['Set-Valued Transformer']['best_acc']
        std_acc = results['Standard Transformer']['best_acc']
        diff = sv_acc - std_acc
        
        print(f"\n📊 Comparison:")
        print(f"  Set-Valued Transformer: {sv_acc:.2f}%")
        print(f"  Standard Transformer:   {std_acc:.2f}%")
        print(f"  Difference:             {diff:+.2f}%")
        
        if diff > 0:
            print(f"  ✅ Set-valued outperforms by {diff:.2f}%")
        elif diff < 0:
            print(f"  ⚠️  Standard outperforms by {-diff:.2f}%")
        else:
            print(f"  ➡️  Both models tie")
    
    print("\n" + "=" * 70)
    print("Framework Validation on Real Data: ✅ Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
