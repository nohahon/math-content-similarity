from torch.utils.data import DataLoader, Dataset
from config import RANDOM_SEED


class TorchDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Access a row from the DataFrame by index
        sample = self.features[index]
        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]
        labels = self.labels[index]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def get_dataloader(self, batch_size):
        return DataLoader(self, batch_size=batch_size, shuffle=True)
