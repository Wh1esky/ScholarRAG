"""
Router 训练脚本

训练 MLP Router 使用准备好的训练数据

使用方法:
    python src/chunking/train_router.py
"""

import os
import json
import torch
from pathlib import Path
from typing import List, Dict, Optional
import sys

import numpy as np
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.chunking.mlp_router import (
    MoGRouter, 
    MoGRouterTrainer, 
    SoftLabelBuilder,
    RouterDataset
)


class RouterTrainingPipeline:
    """
    Router 训练流程
    """
    
    def __init__(
        self,
        model_dir: str = "src/chunking/models",
        data_dir: str = "src/chunking/training_data"
    ):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.router = None
        self.trainer = None
    
    def setup_router(self, device: Optional[str] = None):
        """初始化 Router"""
        print("Initializing MoG Router...")
        self.router = MoGRouter(
            embedding_model='stsb-roberta-large',
            output_dim=4,
            device=device,
            cache_folder='./models'
        )
        self.trainer = MoGRouterTrainer(self.router)
        return self.router
    
    def load_data(self) -> tuple:
        """加载训练和验证数据"""
        train_path = self.data_dir / "router_train_data.json"
        val_path = self.data_dir / "router_val_data.json"
        
        if not train_path.exists():
            raise FileNotFoundError(
                f"Training data not found: {train_path}\n"
                f"Please run: python src/chunking/prepare_training_data.py"
            )
        
        print(f"Loading training data from {train_path}...")
        train_dataset = RouterDataset(str(train_path))
        print(f"  Train samples: {len(train_dataset)}")
        
        val_dataset = None
        if val_path.exists():
            val_dataset = RouterDataset(str(val_path))
            print(f"  Val samples: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        epochs: int = 50,
        batch_size: int = 16,
        lr: float = 1e-4,
        save_best: bool = True
    ):
        """
        训练 Router
        
        Args:
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            epochs: 训练轮数
            batch_size: 批大小
            lr: 学习率
            save_best: 是否保存最佳模型
        """
        if not self.trainer:
            self.setup_router()
        
        # 创建 DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
        
        # 训练循环
        print(f"\n{'='*50}")
        print(f"Starting training...")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {lr}")
        print(f"{'='*50}\n")
        
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # 训练
            train_loss = self.trainer.train_epoch(train_loader, show_progress=True)
            history['train_loss'].append(train_loss)
            
            # 验证
            if val_loader:
                metrics = self.trainer.evaluate(val_loader)
                val_loss = metrics['loss']
                val_acc = metrics['accuracy']
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")
                
                # 保存最佳模型
                if save_best and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save("router_best.pt")
                    print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")
            else:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}")
                
                # 每5个epoch保存一次
                if (epoch + 1) % 5 == 0:
                    self.save(f"router_epoch_{epoch+1}.pt")
            
            # 早停
            if epoch > 20 and val_dataset:
                recent_losses = history['val_loss'][-10:]
                if all(l > min(recent_losses) for l in recent_losses[-3:]):
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        # 保存最后一个模型
        self.save("router_final.pt")
        
        # 保存训练历史
        with open(self.model_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        return history
    
    def save(self, filename: str):
        """保存模型"""
        path = self.model_dir / filename
        self.trainer.save(str(path))
    
    def load(self, filename: str):
        """加载模型"""
        path = self.model_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        if not self.router:
            self.setup_router()
        self.trainer.load(str(path))
    
    def predict(self, query: str) -> Dict:
        """预测"""
        if not self.router:
            raise RuntimeError("Router not loaded. Call load() first.")
        return self.router.predict(query)
    
    def interactive_test(self):
        """交互式测试"""
        if not self.router:
            print("No model loaded. Please train or load a model first.")
            return
        
        print("\n" + "="*50)
        print("Interactive Testing (type 'quit' to exit)")
        print("="*50)
        
        while True:
            query = input("\nEnter query: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            if not query:
                continue
            
            result = self.router.predict(query)
            best_gran = max(result, key=result.get)
            
            print(f"\nQuery: {query}")
            print(f"Weights: {result}")
            print(f"Best granularity: {best_gran} ({result[best_gran]:.2f})")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train MoG Router")
    parser.add_argument("--data-dir", "-d", default="src/chunking/training_data",
                       help="Training data directory")
    parser.add_argument("--model-dir", "-m", default="src/chunking/models",
                       help="Model output directory")
    parser.add_argument("--epochs", "-e", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--test", "-t", action="store_true",
                       help="Interactive test mode")
    parser.add_argument("--load", "-l", default=None,
                       help="Load existing model")
    
    args = parser.parse_args()
    
    pipeline = RouterTrainingPipeline(
        model_dir=args.model_dir,
        data_dir=args.data_dir
    )
    
    if args.load:
        # 加载已有模型
        pipeline.load(args.load)
        if args.test:
            pipeline.interactive_test()
    else:
        # 训练新模型
        try:
            pipeline.setup_router()
            train_dataset, val_dataset = pipeline.load_data()
            history = pipeline.train(
                train_dataset,
                val_dataset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr
            )
            
            print(f"\n{'='*50}")
            print("Training complete!")
            print(f"Model saved to: {pipeline.model_dir}")
            
            # 交互式测试
            print("\nWould you like to test the model? (y/n)")
            if input().strip().lower() == 'y':
                pipeline.interactive_test()
                
        except FileNotFoundError as e:
            print(f"\nError: {e}")
            print("\nPlease prepare training data first:")
            print("  python src/chunking/prepare_training_data.py")


if __name__ == "__main__":
    main()
