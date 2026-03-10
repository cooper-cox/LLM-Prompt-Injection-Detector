from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from ..data import PromptDataset

def train(args):

    breakpoint()    

    # load data from dataset
    data = PromptDataset()

    breakpoint()
    # Use term frequency to transform prompts to TF-IDF

    vectorizer = TfidfVectorizer(
        max_features=50000,     # limit vocabulary
        ngram_range=(1,2),      # unigrams + bigrams
        stop_words="english"
    )

    breakpoint()

    x_train = vectorizer.fit_transform(data.train_df["prompt"])
    x_dev   = vectorizer.transform(data.dev_df["prompt"])
    x_test  = vectorizer.transform(data.test_df["prompt"])

    breakpoint()

    y_train = data.train_df["label"]
    y_dev   = data.dev_df["label"]
    y_test  = data.test_df["label"]

    
    breakpoint()

    # load model from sklearn
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"  # important if dataset is imbalanced
    )

    breakpoint()

    # .fit handles the training loop and updating model weights
    model.fit(x_train, y_train)

    breakpoint()

    dev_probs = model.predict_proba(x_dev)[:,1]
    dev_preds = (dev_probs >= args.threshold).astype(int)

    breakpoint()
    dev_acc = (((dev_preds + y_dev) + 1) % 2).count(1) / len(dev_preds) # if 0 then incorrect if 1 then correct

    #if args.test:
    #    test_preds = model.predict_proba(x_test)[:,1]
