import pandas as pd

"""
    Custom prompt injection dataset class. Targets are 1 for prompt injection 0 for benign prompt.
    Arguments: Str, path to dataset csv
"""
class PromptDataset(Dataset):

    def __init__(self, dataset_csv, split):
        df = pd.read_csv(dataset_csv)
        self.data = df[df["split"] == split]
        self.prompts = self.data["prompts"]
        self.targets = self.data["targets"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.prompts[idx], self.targets[idx])
