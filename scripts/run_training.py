#!/usr/bin/env python
"""
Run training pipeline from config with wandb logging.

Usage:
    python scripts/run_training.py baseline
    python scripts/run_training.py baseline --force-retokenize
    python scripts/run_training.py baseline --wandb-project my-project
"""

import argparse
import yaml
from pathlib import Path
import sys
from datetime import datetime
import torch
import importlib

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.dataset_tokenizer import DatasetTokenizer
from utils.dataset_encoder import DatasetEncoder
from utils.classifier_trainer import ClassifierTrainer
from transformers import AutoModel


class TrainingPipeline:
    """Training pipeline with automatic caching and wandb logging"""
    
    def __init__(self, experiment_config_name: str, wandb_logger=None, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config = self._load_config(experiment_config_name)
        self.wandb = wandb_logger
        
    def _load_config(self, experiment_name: str) -> dict:
        """Load and merge all config components"""
        exp_path = self.config_dir / "experiments" / f"{experiment_name}.yaml"
        config = yaml.safe_load(exp_path.open())
        
        # Merge component configs
        for component in ['tokenizer', 'encoder', 'classifier']:
            component_path = self.config_dir / f"{component}s" / f"{config[component]}.yaml"
            config[f"{component}_config"] = yaml.safe_load(component_path.open())
        
        return config
    
    def _get_tokenized_path(self) -> Path:
        """Get path for tokenized dataset"""
        dataset_name = self.config['dataset']['name']
        tokenizer_name = self.config['tokenizer_config']['name']
        return Path(self.config['paths']['processed_data']) / dataset_name / "tokenized" / tokenizer_name
    
    def _get_encoded_path(self) -> Path:
        """Get path for encoded dataset"""
        dataset_name = self.config['dataset']['name']
        tok = self.config['tokenizer_config']['name'].split('-')[0]
        enc = self.config['encoder_config']['name'].split('-')[0]
        return Path(self.config['paths']['processed_data']) / dataset_name / "encoded" / f"{tok}_{enc}"
    
    def _is_cached(self, path: Path) -> bool:
        """Check if dataset is cached"""
        return path.exists() and (path / "dataset_dict.json").exists()
    
    def run_tokenization(self, force: bool = False) -> Path:
        """Step 1: Tokenize dataset (with caching)"""
        tokenized_path = self._get_tokenized_path()
        
        if not force and self._is_cached(tokenized_path):
            print(f"✓ Using cached tokenized data: {tokenized_path}")
            if self.wandb:
                self.wandb.log({"tokenization": "cached"})
            return tokenized_path
        
        print(f"\n=== Tokenizing dataset ===")
        raw_csv = Path(self.config['paths']['raw_data']) / self.config['dataset']['file']
        
        tokenizer = DatasetTokenizer(
            tokenizer_name=self.config['tokenizer_config']['name'],
            output_dir=str(tokenized_path.parent.parent),
            max_length=self.config['tokenizer_config']['max_length'],
            test_size=self.config['dataset']['test_size'],
            random_state=self.config['random_state']
        )
        
        tokenizer.output_dir = tokenized_path.parent
        
        tokenizer.tokenize_and_save(
            csv_path=str(raw_csv),
            text_column=self.config['dataset']['text_column'],
            label_column=self.config['dataset']['label_column'],
            batch_size=self.config['tokenizer_config']['batch_size']
        )
        
        # Move to final location if needed
        default_path = tokenized_path.parent / "tokenized_dataset"
        if default_path.exists() and default_path != tokenized_path:
            import shutil
            if tokenized_path.exists():
                shutil.rmtree(tokenized_path)
            shutil.move(str(default_path), str(tokenized_path))
        
        if self.wandb:
            self.wandb.log({
                "tokenization": "completed",
                "tokenizer": self.config['tokenizer_config']['name'],
                "max_length": self.config['tokenizer_config']['max_length']
            })
        
        return tokenized_path
    
    def run_encoding(self, tokenized_path: Path, force: bool = False) -> Path:
        """Step 2: Encode dataset (with caching)"""
        encoded_path = self._get_encoded_path()
        
        if not force and self._is_cached(encoded_path):
            print(f"✓ Using cached encoded data: {encoded_path}")
            if self.wandb:
                self.wandb.log({"encoding": "cached"})
            return encoded_path
        
        print(f"\n=== Encoding dataset ===")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        encoder_model = AutoModel.from_pretrained(self.config['encoder_config']['name'])
        
        dataset_encoder = DatasetEncoder(
            encoder=encoder_model,
            output_dir=str(encoded_path),
            device=device
        )
        
        dataset_encoder.encode_and_save(
            tokenized_dataset_path=str(tokenized_path),
            batch_size=self.config['encoder_config']['batch_size']
        )
        
        if self.wandb:
            self.wandb.log({
                "encoding": "completed",
                "encoder": self.config['encoder_config']['name'],
                "device": device
            })
        
        return encoded_path
    
    def _instantiate_classifier(self):
        """Dynamically create classifier from config"""
        clf_config = self.config['classifier_config']
        module_path, class_name = clf_config['type'].rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)(**clf_config['params'])
    
    def run_training(self, encoded_path: Path):
        """Step 3: Train classifier"""
        print(f"\n=== Training classifier ===")
        
        classifier = self._instantiate_classifier()
        
        # Pass wandb logger to trainer
        trainer = ClassifierTrainer(
            head=classifier,
            model_save_path=None,  # Don't save to disk
            wandb_logger=self.wandb
        )
        
        trainer.load_encoded_dataset(str(encoded_path))
        trainer.train(
            epochs=self.config['training']['epochs'],
            batch_size=self.config['training']['batch_size'],
            eval_every=self.config['training']['eval_every']
        )
        
        print(f"\n✓ Training complete!")
        return trainer
    
    def run(self, force_retokenize: bool = False, force_reencode: bool = False):
        """Run full pipeline with smart caching"""
        print(f"=== Running training: {self.config['experiment_name']} ===\n")
        print(f"Pipeline: {self.config['tokenizer']} → {self.config['encoder']} → {self.config['classifier']}")
        
        # Log config to wandb
        if self.wandb:
            self.wandb.config.update(self.config)
        
        # Step 1: Tokenization
        tokenized_path = self.run_tokenization(force=force_retokenize)
        
        # Step 2: Encoding
        encoded_path = self.run_encoding(tokenized_path, force=force_reencode)
        
        # Step 3: Training
        trainer = self.run_training(encoded_path)
        
        return trainer


def main():
    parser = argparse.ArgumentParser(
        description="Run text classification training with automatic caching"
    )
    parser.add_argument('experiment', type=str, help='Experiment config name (e.g., baseline)')
    parser.add_argument('--force-retokenize', action='store_true', 
                       help='Force re-tokenization even if cached')
    parser.add_argument('--force-reencode', action='store_true',
                       help='Force re-encoding even if cached')
    parser.add_argument('--wandb-project', type=str, default=None,
                       help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                       help='Weights & Biases entity (username/team)')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                       help='Weights & Biases run name')
    args = parser.parse_args()
    
    # Initialize wandb if project specified
    wandb_logger = None
    if args.wandb_project:
        try:
            import wandb
            wandb_logger = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                config={}  # Will be updated with full config in pipeline
            )
            print(f"✓ Weights & Biases logging enabled: {args.wandb_project}")
        except ImportError:
            print("⚠ wandb not installed. Install with: pip install wandb")
            print("  Continuing without wandb logging...")
    
    pipeline = TrainingPipeline(args.experiment, wandb_logger=wandb_logger)
    pipeline.run(
        force_retokenize=args.force_retokenize,
        force_reencode=args.force_reencode
    )
    
    if wandb_logger:
        wandb_logger.finish()


if __name__ == "__main__":
    main()

