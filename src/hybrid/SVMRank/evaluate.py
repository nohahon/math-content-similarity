import argparse
import subprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        required=False,
        default="test",
    )
    parser.add_argument(
        "--rand",
        type=bool,
        required=False,
        default=False,
    )
    args = parser.parse_args()

    split = args.split
    rand = args.rand

    # train the re-ranker
    command = [
        "./svm_rank_learn",
        "-c",
        "5",
        "./data/train/train.dat",
        "./data/train/model_trained.dat",
    ]
    result = subprocess.run(command, capture_output=True)

    if result.returncode != 0:
        print("Training failed.")
        print(result.stdout)
        exit(1)

    # generate predictions
    command = [
        "./svm_rank_classify",
        f"./data/{split}/{split}.dat",
        "./data/train/model_trained.dat",
    ]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        print("Inference failed.")
        print(result.stdout)
        exit(1)

    # evaluate F1
    result = subprocess.run(
        ["python", "./scripts/eval_predictions.py", "--split", f"{split}"],
        capture_output=False,
    )
