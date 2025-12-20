"""
Activation extractor for BERT layers.
Extracts hidden states from specified transformer layers for interpretability analysis.
"""

import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import List, Dict


class ActivationExtractor:
    """
    Extracts activations from specified BERT layers.

    Args:
        encoder: The encoder model (BERT)
        layers: List of layer indices to extract (e.g., [3, 6, 9, 12])
        device: Device to run on ('cuda' or 'cpu')
    """

    def __init__(
        self,
        encoder,
        layers: List[int],
        device: str = "cuda"
    ):
        self.encoder = encoder
        self.layers = layers
        self.device = device
        self.encoder.to(device)
        self.encoder.eval()

        # Enable output of hidden states
        self.encoder.config.output_hidden_states = True

        print(f"Activation extractor initialized")
        print(f"  Layers to extract: {layers}")
        print(f"  Device: {device}")

    def extract_activations(
        self,
        dataset: Dataset,
        batch_size: int = 64,
        max_samples: int = None,
        desc: str = "Extracting activations"
    ) -> Dict[int, np.ndarray]:
        """
        Extract activations from specified layers with balanced class sampling.
        
        Args:
            dataset: Tokenized dataset with 'input_ids', 'attention_mask', 'label'
            batch_size: Batch size for processing
            max_samples: Maximum number of samples to process (None = all), will be balanced 50/50
            desc: Progress bar description

        Returns:
            Dictionary mapping layer_idx -> activations array (N_samples, 768)
            Labels array (N_samples,)
        """
        from datasets import concatenate_datasets
        
        # #region agent log
        import json
        with open(r'd:\_gradient\Human-vs-AI-text-Text-Classification-Analysis-\.cursor\debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"initial","hypothesisId":"A,B,C","location":"activation_extractor.py:67","message":"Dataset info before filtering","data":{"column_names":dataset.column_names,"features":str(dataset.features),"num_rows":len(dataset),"max_samples":max_samples},"timestamp":int(__import__('time').time()*1000)})+'\n')
        # #endregion
        
        # Balance dataset if max_samples is specified
        if max_samples:
            # #region agent log
            with open(r'd:\_gradient\Human-vs-AI-text-Text-Classification-Analysis-\.cursor\debug.log', 'a') as f:
                sample_item = dataset[0]
                f.write(json.dumps({"sessionId":"debug-session","runId":"initial","hypothesisId":"A,D","location":"activation_extractor.py:75","message":"First dataset item keys","data":{"keys":list(sample_item.keys()),"has_label":('label' in sample_item),"has_labels":('labels' in sample_item)},"timestamp":int(__import__('time').time()*1000)})+'\n')
            # #endregion
            
            # Split by class
            ai_samples = dataset.filter(lambda x: x['label'] == 1)
            human_samples = dataset.filter(lambda x: x['label'] == 0)
            
            # Calculate balanced split
            samples_per_class = max_samples // 2
            
            # Take equal numbers from each class
            ai_subset = ai_samples.shuffle(seed=42).select(range(min(samples_per_class, len(ai_samples))))
            human_subset = human_samples.shuffle(seed=42).select(range(min(samples_per_class, len(human_samples))))
            
            # Combine and shuffle
            dataset = concatenate_datasets([ai_subset, human_subset]).shuffle(seed=42)
            
            # #region agent log
            with open(r'd:\_gradient\Human-vs-AI-text-Text-Classification-Analysis-\.cursor\debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"initial","hypothesisId":"C","location":"activation_extractor.py:95","message":"After balanced sampling","data":{"ai_count":len(ai_subset),"human_count":len(human_subset),"total":len(dataset),"column_names":dataset.column_names},"timestamp":int(__import__('time').time()*1000)})+'\n')
            # #endregion
            
            print(f"Balanced sampling: {len(ai_subset)} AI + {len(human_subset)} Human = {len(dataset)} total")
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Initialize storage for each layer
        layer_activations = {layer: [] for layer in self.layers}
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                # #region agent log
                if 'batch_logged' not in locals():
                    with open(r'd:\_gradient\Human-vs-AI-text-Text-Classification-Analysis-\.cursor\debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"initial","hypothesisId":"D","location":"activation_extractor.py:108","message":"First batch keys","data":{"batch_keys":list(batch.keys()),"has_label":('label' in batch),"has_labels":('labels' in batch)},"timestamp":int(__import__('time').time()*1000)})+'\n')
                    batch_logged = True
                # #endregion
                
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"]

                # Get encoder output with hidden states
                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )

                # Extract CLS token activations from specified layers
                # hidden_states is tuple of (embeddings, layer1, layer2, ..., layer12)
                # Index 0 is embeddings, layers are 1-indexed
                for layer_idx in self.layers:
                    # Get hidden state for this layer (batch_size, seq_len, hidden_size)
                    layer_output = outputs.hidden_states[layer_idx]
                    # Extract CLS token (position 0)
                    cls_activations = layer_output[:, 0, :]
                    layer_activations[layer_idx].append(cls_activations.cpu().numpy())

                all_labels.append(labels.numpy())

        # Concatenate all batches
        for layer_idx in self.layers:
            layer_activations[layer_idx] = np.vstack(layer_activations[layer_idx])

        labels_array = np.concatenate(all_labels)

        print(f"\n✓ Extracted activations from {len(self.layers)} layers")
        print(f"  Shape per layer: {layer_activations[self.layers[0]].shape}")
        print(f"  Total samples: {len(labels_array)}")

        return layer_activations, labels_array

    def extract_and_save(
        self,
        tokenized_dataset_path: str,
        output_dir: str,
        batch_size: int = 64,
        max_samples: int = None,
        split: str = "train"
    ):
        """
        Extract activations and save to disk.

        Args:
            tokenized_dataset_path: Path to tokenized dataset
            output_dir: Directory to save activations
            batch_size: Batch size for extraction
            max_samples: Maximum samples to process
            split: Dataset split to use ('train' or 'test')
        """
        from datasets import load_from_disk

        print(f"Loading tokenized dataset from {tokenized_dataset_path}...")
        dataset_dict = load_from_disk(tokenized_dataset_path)
        
        # #region agent log
        import json
        with open(r'd:\_gradient\Human-vs-AI-text-Text-Classification-Analysis-\.cursor\debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"initial","hypothesisId":"E","location":"activation_extractor.py:155","message":"Dataset dict loaded","data":{"splits":list(dataset_dict.keys()),"requested_split":split},"timestamp":int(__import__('time').time()*1000)})+'\n')
        # #endregion
        
        dataset = dataset_dict[split]

        print(f"\nExtracting activations from {split} split...")
        activations, labels = self.extract_activations(
            dataset=dataset,
            batch_size=batch_size,
            max_samples=max_samples,
            desc=f"Extracting {split}"
        )

        # Save activations
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for layer_idx, layer_acts in activations.items():
            save_path = output_path / f"layer_{layer_idx}_activations.npy"
            np.save(save_path, layer_acts)
            print(f"  Saved layer {layer_idx}: {save_path}")

        # Save labels
        labels_path = output_path / "labels.npy"
        np.save(labels_path, labels)
        print(f"  Saved labels: {labels_path}")

        # Save metadata
        metadata = {
            "layers": self.layers,
            "n_samples": len(labels),
            "n_ai": int((labels == 1).sum()),
            "n_human": int((labels == 0).sum()),
            "split": split,
            "activation_shape": activations[self.layers[0]].shape
        }

        import json
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved metadata: {metadata_path}")

        print(f"\n✓ All activations saved to {output_path}")


