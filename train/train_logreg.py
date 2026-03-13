from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from data.dataset import PromptDataset
from sklearn.metrics import confusion_matrix
import logging
import joblib

def evaluate(model, x, y, config):
    probabilities = model.predict_proba(x)[:,1]
    predictions = (probabilities >= config.threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()
    #logger.log
    print(f"tp: {tp}\nfp: {fp}\nfn: {fn}\ntn:{tn}\n")

    acc = (tn + tp) / len(y)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = (2 * precision * recall) / (precision + recall)

    return acc, precision, recall, f1

def train(args):
    logger = logging.getLogger(__name__)

    logger.info("Loading train and eval datasets")
    
    # load data from dataset
    train_dataset = PromptDataset(args.data_path, "train")
    if args.test:
        eval_dataset = PromptDataset(args.data_path, "test")
    else:
        eval_dataset = PromptDataset(args.data_path, "dev")

    # Use term frequency to transform prompts to TF-IDF

    vectorizer = TfidfVectorizer(
        max_features=50000,     # limit vocabulary
        ngram_range=(1,2),      # unigrams + bigrams
        stop_words="english"
    )

    logger.info("TF-IDF encoding prompts")
    x_train = vectorizer.fit_transform(train_dataset.prompts)
    x_eval = vectorizer.transform(eval_dataset.prompts)

    y_train = train_dataset.targets
    y_eval = eval_dataset.targets
    
    # load model from sklearn
    logger.info("Loading logistic regression model")
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"  # important if dataset is imbalanced
    )

    # .fit handles the training loop and updating model weights
    logger.info("Training model")
    model.fit(x_train, y_train)

    logger.info("Evaluating model performance")
    train_acc, train_precision, train_recall, train_f1 = evaluate(model, x_train, y_train, args)
    eval_acc, eval_precision, eval_recall, eval_f1 = evaluate(model, x_eval, y_eval, args)
    logger.info(f"TRAIN ACC: {train_acc}")
    logger.info(f"TRAIN PRECISION: {train_precision}")
    logger.info(f"TRAIN RECALL: {train_recall}")
    logger.info(f"TRAIN F1: {train_f1}")
    logger.info(f"{'TEST' if args.test else 'DEV'} ACC: {eval_acc}")
    logger.info(f"{'TEST' if args.test else 'DEV'} PRECISION: {eval_precision}")
    logger.info(f"{'TEST' if args.test else 'DEV'} RECALL: {eval_recall}")
    logger.info(f"{'TEST' if args.test else 'DEV'} F1: {eval_f1}")

    args.save_path.mkdir(parents=True, exist_ok=True)
    logger.info("Saving model weights and TF-IDF vectorizer")
    joblib.dump(vectorizer, f"{args.save_path}/tfidf.pkl")
    joblib.dump(model, f"{args.save_path}/model.pkl")
