import argparse
from pathlib import Path
import logger
import time

def parse_all_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-D", "--dataset", help = "Whether or not to create a new dataset (bool) [default = False]", default=False, type=bool)
    parser.add_argument("-M", "--model", help = "Which model architecture to use (str) [options: (logreg, bert, large-bert] [default=logreg]", default="logreg", type=str)
    parser.add_argument("-t", "--test", help = "Whether or not to train a new model or evaluate on test split (bool) [default=False]", default=False, type=bool)
    parser.add_argument("-d", "--data_path", help = "Path to dataset csv file (Path) [default=repo_dir/data/dataset/dataset.csv]", default=Path(__file__).resolve().parent / "data" / "dataset" / "dataset.csv", type=Path)
    parser.add_argument("-th", "--threshold", help = "Model threshold to predict malicious prompt (float) [default = 0.5]", default=0.5, type=float)
    parser.add_argument("-e", "--epoch", help = "Number of epochs (int) [default=10]", default=10, type=int)
    parser.add_argument("-b", "--batch", help = "Batch size for bert models (int) [default=32]", default=32, type=int)
    parser.add_argument("-seq", "--seq_length", help = "max sequence length of prompts (int) [default=128]", default=128, type=int)
    parser.add_argument("-lr", "--lr", help = "optimizer learning rate (float) [default=0.00001]", default=0.00001, type=float)
    parser.add_argument("-s", "--save_path", help = "path to save model weights (str) [default=repo_dir/model/weights/]", default=Path(__file__).resolve().parent / "model" / "weights", type=Path)

    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")

    return parser.parse_args()

import logging

def setup_logger(log_file):
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Format
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main():
    args = parse_all_args()
    logger = setup_logger(f"logs/train_{args.model}_{str(args.threshold).replace('.', '_')}.log")

    # DATASET
    if args.dataset:
        from data.load_and_save import create_dataset
        logger.info("Creating new dataset")
        create_dataset()

    # TRAINING
    if args.model == "logreg":
        from train.train_logreg import train
        args.save_path = args.save_path / "logreg" / str(args.threshold).replace('.', '_')
        logger.info("Training new logistic regression model")
        train(args)
    elif args.model == "bert":
        from train.train_bert import train
        args.save_path = args.save_path / "bert"
        logger.info("Training new finetuned BERT model")
        train(args)
    else:
        logger.error("Model architecture specified is not currently supported. Supported model architectures: [logreg, bert].") 
        raise ValueError("Model architecture not among currently available model options.")

if __name__ == "__main__":
    main()
