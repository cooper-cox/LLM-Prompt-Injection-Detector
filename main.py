import argparse
from pathlib import Path

def parse_all_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", help = "Which model architecture to use (str) [options: (logreg, bert] [default=logreg]", default="logreg", type=str)
    parser.add_argument("-t", "--test", help = "Whether or not to train a new model or evaluate on test split (bool) [default=False]", default=False, type=bool)
    parser.add_argument("-d", "--data", help = "Path to dataset csv file (Path) [default=path/to/dataset.csv]", default=Path(__file__).resolve().parent / "data" / "dataset" / "dataset.csv", type=Path)
    parser.add_argument("-th", "--threshold", help = "Model threshold to predict malicious prompt (float) [default = 0.5]", default=0.5, type=float)
    parser.add_argument("-e", "--epoch", help = "Number of epochs (int) [default=10]", default=10, type=int)
    parser.add_argument("-lr", help = "optimizer learning rate (float) [default=0.001]", default=0.001, type=float)
    parser.add_argument("-s", "--save", help = "path to save model weights (str) [default=path/to/weights/dir]", default=Path(__file__).resolve().parent / "model" / "weights", type=Path)

    return parser.parse_args()

def main():
    args = parse_all_args()

    breakpoint()

    if args.model == "logreg":
        from train.train_logreg import train
        args.save = args.save / "logreg"
        breakpoint()
        train(args)
    elif args.model == "bert":
        #from train.train_logreg import train
        args.save = args.save / "bert"
        #breakpoint()
        #train(args)
    else:
        raise ValueError("Model architecture not among currently available model options.")

if __name__ == "__main__":
    main()
