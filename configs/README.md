# Experiment Configuration

This directory contains hierarchical configuration files for running text classification experiments.

## Structure

```
configs/
├── experiments/     # Main experiment configs
├── tokenizers/      # Tokenizer configurations
├── encoders/        # Encoder model configurations
└── classifiers/     # Classifier configurations
```

## Usage

Run training:
```bash
python scripts/run_training.py baseline
```

With Weights & Biases logging:
```bash
python scripts/run_training.py baseline --wandb-project my-project
```

Force re-tokenization or re-encoding:
```bash
python scripts/run_training.py baseline --force-retokenize
python scripts/run_training.py baseline --force-reencode
```

## Adding New Experiments

1. Create a new experiment config in `experiments/`
2. Reference existing or new tokenizer/encoder/classifier configs
3. Run with: `python scripts/run_training.py <your_experiment_name>`

## Caching

The pipeline automatically caches:
- **Tokenized datasets**: `data/processed/{dataset}/tokenized/{tokenizer_name}/`
- **Encoded datasets**: `data/processed/{dataset}/encoded/{tokenizer}_{encoder}/`

Training metrics are logged to Weights & Biases (if enabled) or console output.

