from torch.utils.data import Dataset, DataLoader, Subset
import torch


class EncodedDataset(Dataset):
    def __init__(self, dataset, encoder, device):
        """
        Custom dataset that computes encoder outputs on-demand.
        """
        self.dataset = dataset
        self.encoder = encoder
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        items = self.dataset[idx]
        input_ids: torch.Tensor = items["input_ids"].detach().clone().to(self.device)
        attention_mask: torch.Tensor = (
            items["attention_mask"].detach().clone().to(self.device)
        )

        with torch.no_grad():
            output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            X = output.last_hidden_state.squeeze(0)

        y = items["labels"]
        return X.reshape(len(idx), -1).cpu(), y.type(torch.int64).ravel().cpu()


def get_encoded_dataloader(dataset, encoder, batch_size, device, subset_len=None):
    """
    Creates an instance of EncodedDataset and wraps it inside dataloader.
    Subset of the dataset can be loaded, using subset_len. Will load the
    whole dataset by default
    """
    if subset_len is not None:
        dataset = Subset(dataset, torch.randint(0, len(dataset), (subset_len,)))

    encoded_dataset = EncodedDataset(dataset, encoder, device)
    return DataLoader(encoded_dataset, batch_size, shuffle=True)
