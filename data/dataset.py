import pandas as pd
from torch.utils.data import Dataset

"""
    Custom prompt injection dataset class. Targets are 1 for prompt injection 0 for benign prompt.
    Arguments: Str, path to dataset csv
"""
class PromptDataset(Dataset):

    def __init__(self, dataset_csv, split):
        df = pd.read_csv(dataset_csv, keep_default_na=False)
        self.data = df[df["split"] == split]
        self.prompts = self.data["prompt"]
        self.targets = self.data["target"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.prompts[idx], self.targets[idx])
