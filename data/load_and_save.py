import pandas as pd
import langid
from sklearn.model_selection import train_test_split
import logging

def is_english(text):
    lang, score = langid.classify(text)
    return lang == "en"

def load_qualifire():
    # load
    df = pd.read_csv("hf://datasets/qualifire/prompt-injections-benchmark/test.csv")

    # clean
    df["target"] = df["label"] == "jailbreak"
    df["target"] = df["target"].astype(int)
    df = df.rename(columns={"text": "prompt"})
    cleaned_df = df.drop("label", axis=1)
    
    # remove non english rows
    cleaned_df = cleaned_df[cleaned_df["prompt"].apply(is_english)]

    cleaned_df["split"] = "train"

    return cleaned_df

# problematic unlabeled dataset
def load_224():
    # load
    df = pd.read_csv("hf://datasets/xxz224/prompt-injection-attack-dataset/complete_dataset.csv")

    # clean
    cleaned_df = df

    return cleaned_df

def load_safejay():
    # load
    splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/jayavibhav/prompt-injection-safety/" + splits["train"])

    test_df = pd.read_parquet("hf://datasets/jayavibhav/prompt-injection-safety/" + splits["test"])
    test_df = test_df[test_df["label"] == 0] 
    test_df = test_df.rename(columns={"text": "prompt", "label": "target"})    

    # clean
    df["target"] = df["label"].replace({2:1})
    df = df.rename(columns={"text": "prompt"})
    cleaned_df = df.drop("label", axis=1)
    
    cleaned_df = cleaned_df[cleaned_df["prompt"].apply(is_english)]
    cleaned_test_df = test_df[test_df["prompt"].apply(is_english)]

    cleaned_df["split"] = "test"
    cleaned_test_df["split"] = "train"

    return cleaned_df, cleaned_test_df

def load_bigjay():
    # load
    splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/jayavibhav/prompt-injection/" + splits["train"])
    test_df = pd.read_parquet("hf://datasets/jayavibhav/prompt-injection/" + splits["test"])

    # clean
    cleaned_df = df.rename(columns={"text":"prompt", "label":"target"})
    cleaned_test_df = test_df.rename(columns={"text":"prompt", "label":"target"})

    cleaned_df = cleaned_df[cleaned_df["prompt"].apply(is_english)]
    cleaned_test_df = cleaned_test_df[cleaned_test_df["prompt"].apply(is_english)]

    cleaned_df["split"] = "train"
    cleaned_test_df["split"] = "test"

    return cleaned_df, cleaned_test_df

def create_dataset():
    logger = logging.getLogger(__name__)

    # load
    logger.info("Loading datasets")
    qual = load_qualifire()
    safejay_train, safejay_test_benign = load_safejay()
    bigjay_train, bigjay_test = load_bigjay()
 
    # merge and clean
    logger.info("Merging into a single dataset and cleaning")
    merged_df = pd.concat([qual, safejay_train, safejay_test_benign, bigjay_train, bigjay_test], ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset="prompt", keep="first")
    merged_df["prompt"] = merged_df["prompt"].fillna("").astype(str)
    merged_df = merged_df[merged_df["prompt"].str.strip() != ""]

    # split dataset
    logger.info("Splitting into train/dev/test")
    test_df = merged_df[merged_df["split"] == "test"]
    
    dev_df, test_df = train_test_split(
        test_df,
        test_size=0.5,
        random_state=42,
        stratify=test_df["target"]
    )
   
    dev_df["split"] = "dev"
    train_df = merged_df[merged_df["split"] == "train"]

    merged_df = pd.concat([train_df, dev_df, test_df]).reset_index(drop=True)

    logger.info("Saving dataset to dataset dir")
    merged_df.to_csv("data/dataset/dataset.csv", index=False)
