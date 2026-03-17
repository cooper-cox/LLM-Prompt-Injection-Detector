import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dataset import PromptDataset
from transformers import AutoTokenizer
from model.promptClassifier import PromptClassifier
import logging
import numpy as np
from sklearn.metrics import confusion_matrix
import wandb

def evaluate_model(model, dataloader, device, config):

    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            probs = model(input_ids, attention_mask)

            all_probs.extend(probs.squeeze().cpu().numpy())
            all_labels.extend(labels)

    probabilities = np.array(all_probs)
    y = np.array(all_labels)

    predictions = (probabilities >= config.threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()

    print(f"tp: {tp}\nfp: {fp}\nfn: {fn}\ntn:{tn}\n")

    acc = (tn + tp) / len(y)
    recall = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    f1 = (2 * precision * recall) / (precision + recall + 1e-8)

    model.train()

    return round(acc, 3), round(precision, 3), round(recall, 3), round(f1, 3)


def train(config):
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.use_wandb:
        logger.info("Initializing Wandb")
        wandb.init(
            project="prompt-injection-detector",
            config=vars(config)
        )

    is_sweep = config.use_wandb and wandb.run is not None and wandb.run.sweep_id is not None

    # LOAD TOKENIZER
    logger.info("Loading model tokenizer")
    if config.model == "bert":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    elif config.model == "bert-large":
        tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

    breakpoint()
   
    # LOAD DATASETS AND DATALOADERS
    logger.info("Loading train and eval datasets/dataloaders")
    train_dataset = PromptDataset("train", config, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config.batch, shuffle=True)
   
    breakpoint()

    if config.test:
        eval_dataset = PromptDataset("test", config, tokenizer)
    else:
        eval_dataset = PromptDataset("dev", config, tokenizer)
    
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch, shuffle=False)

    
    breakpoint()

    model = PromptClassifier(config).to(device)

    breakpoint()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    criterion = nn.BCELoss()

    # TODO: Fix train loop, get accuracy/evaluation metric, save model weights 

    model.train()


    breakpoint()

    # TRAIN LOOP
    logger.info("Beginning train loop")
    for epoch in range(config.epoch):
        # TRAIN
        epoch_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            probs = model(input_ids, attention_mask)

            loss = criterion(probs.squeeze(), labels)

            epoch_loss += loss.item()

            loss.backward()

            optimizer.step()

        logger.info(f"Epoch {epoch+1} finished")
        
        if config.use_wandb:
            wandb.log({"train_loss": epoch_loss / len(train_loader)})
       

        # EVALUATE
        logger.info("Evaluating on TRAIN set")

        train_acc, train_prec, train_rec, train_f1 = evaluate_model(model, train_loader, device, config)

        if config.use_wandb:
            wandb.log({
                "train_accuracy": train_acc,
                "train_precision": train_prec,
                "train_recall": train_rec,
                "train_f1": train_f1
            })


        logger.info(f"TRAIN acc:{train_acc} prec:{train_prec} rec:{train_rec} f1:{train_f1}")

        # DEV/TEST
        if (epoch + 1) % 5 == 0:
            
            logger.info("Evaluating on EVAL set")
            
            eval_acc, eval_prec, eval_rec, eval_f1 = evaluate_model(model, eval_loader, device, config)

            if config.use_wandb:
                wandb.log({
                    "eval_accuracy": eval_acc,
                    "eval_precision": eval_prec,
                    "eval_recall": eval_rec,
                    "eval_f1": eval_f1
                })

            logger.info(f"EVAL acc:{eval_acc} prec:{eval_prec} rec:{eval_rec} f1:{eval_f1}")


    breakpoint()

    # SAVE MODEL (skip during sweeps)
    if not is_sweep:

        logger.info("Saving model")

        config.save_path.mkdir(parents=True, exist_ok=True)

        model_path = config.save_path / "bert_model.pt"

        torch.save(model.state_dict(), model_path)

        logger.info(f"Model saved to {model_path}")

    else:

        logger.info("Skipping model save (sweep run)")
