#!/usr/bin/env python
"""
Train all classifier models and save them to models/ directory.

This trains 5 classifiers on the same BERT embeddings:
- Linear SVC
- SGD Classifier
- Random Forest
- Logistic Regression
- Decision Tree

Models are saved as: models/{classifier_name}.pkl

Usage:
    python scripts/train_all_classifiers.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.training_pipeline import TrainingPipeline


def main():
    experiments = [
        "sgd",           # → models/sgd.pkl
        "linear_svc",         # → models/linear_svc.pkl
        "random_forest",      # → models/random_forest.pkl
        "logistic_regression", # → models/logistic_regression.pkl
        "decision_tree",      # → models/decision_tree.pkl
    ]
    
    print("=" * 80)
    print(f"Training and saving {len(experiments)} classifier models")
    print("=" * 80)

    results = {}
    
    for i, exp_name in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Training: {exp_name}")
        try:
            pipeline = TrainingPipeline(exp_name)
            pipeline.run(save_model=True)
            results[exp_name] = "Success"
        except Exception as e:
            print(f"✗ {exp_name} failed: {e}")
            results[exp_name] = f"✗ Failed: {e}"
    
    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    for exp_name, status in results.items():
        print(f"{exp_name:25s} {status}")
    print("=" * 80)
    print("\nModels saved to models/ directory")
    print("  Load them with AiHumanPredictor in notebooks!")


if __name__ == "__main__":
    main()

