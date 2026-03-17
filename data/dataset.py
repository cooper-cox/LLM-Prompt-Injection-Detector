import pandas as pd
import torch
from torch.utils.data import Dataset

"""
    Custom prompt injection dataset class. Targets are 1 for prompt injection 0 for benign prompt.
    Arguments: Str, path to dataset csv
"""
class PromptDataset(Dataset):

    def __init__(self, split, config, tokenizer=None):
        df = pd.read_csv(config.data_path, keep_default_na=False)
        self.data = df[df["split"] == split].reset_index(drop=True)
        self.prompts = self.data["prompt"].tolist()
        self.targets = self.data["target"].tolist()
        self.model = config.model

        if tokenizer:
            self.encodings = tokenizer(
                self.prompts,
                padding=True,
                truncation=True,
                max_length=config.seq_length,
                return_tensors="pt"
            )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if self.model == "logreg":
            return (self.prompts[idx], self.targets[idx])
        else:
            item = {key: val[idx] for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.targets[idx], dtype=torch.float)

            # dict with keys: input_ids, attention_mask, labels
            return item
