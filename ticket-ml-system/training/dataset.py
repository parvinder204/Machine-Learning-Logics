from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class TicketDataset(Dataset):
    def __init__(self, split_path: str | Path):
        data = torch.load(split_path, weights_only=True)
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.labels = data["labels"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

def get_dataloaders(
    processed_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    processed_dir = Path(processed_dir)

    train_ds = TicketDataset(processed_dir / "train.pt")
    val_ds = TicketDataset(processed_dir / "val.pt")
    test_ds = TicketDataset(processed_dir / "test.pt")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader