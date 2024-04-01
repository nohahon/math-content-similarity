import subprocess
import itertools
from tqdm import tqdm
import pandas as pd


def train_model(c, p, o, w, t, n_dim):
    """
    Trains a model using SVM-Rank algorithm.

    This function trains a model using the SVM-Rank algorithm by executing the 'svm_rank_learn' command-line tool.
    It takes the following parameters:
    - c: The trade-off parameter for training.
    - p: The epsilon parameter for training.
    - o: The loss function option for training.
    - w: The weight option for training.
    - t: The kernel type for training.
    - n_dim: The dimension of the data.

    Parameters:
        c (int): The trade-off parameter for training.
        p (int): The epsilon parameter for training.
        o (int): The loss function option for training.
        w (int): The weight option for training.
        t (int): The kernel type for training.
        n_dim (int): The dimension of the data.

    Returns:
        int: The return code of the 'svm_rank_learn' command-line tool.
    """
    command = [
        "./svm_rank_learn",
        "-c",
        str(c),
        "-p",
        str(p),
        "-o",
        str(o),
        "-w",
        str(w),
        "-t",
        str(t),
        "./data/train/train.dat",
        "./data/train/model_trained.dat",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )
    return result.returncode


def predict():
    """
    Makes predictions using the trained model.

    This function makes predictions using the trained model by executing the 'svm_rank_classify' command-line tool.

    Parameters:
        None

    Returns:
        int: The return code of the 'svm_rank_classify' command-line tool.
    """
    command = [
        "./svm_rank_classify",
        "./data/dev/dev.dat",
        "./data/train/model_trained.dat",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=False,
    )
    return result.returncode


def clean_data():
    """
    Cleans the data by removing the trained model file and svm_predictions file.

    This function removes the 'model_trained.dat' file located in the 'data/train' directory
    and the 'svm_predictions' file from the current working directory.

    Parameters:
        None

    Returns:
        None
    """
    subprocess.run(
        ["rm", "data/train/model_trained.dat", "svm_predictions"],
        capture_output=False,
        text=False,
    )


param_dict = {
    "-c": [3, 10, 50],
    "-p": [1, 2],
    "-o": [1, 2],
    "-w": [0, 1, 2, 3],
    "-t": [1, 2, 3],
    "n_dim": [20, 200],
}

combinations = list(itertools.product(*param_dict.values()))
row_lines = []
latest = 20
for comb in tqdm(combinations):
    # dimension change
    if comb[5] != latest:
        res = subprocess.run(
            ["python", "create_data.py", "--n_dim", f"{comb[5]}"],
            capture_output=True,
            text=True,
        )
        latest = comb[5]

    # train the model
    failed = train_model(*comb)

    if failed:
        continue

    # make predictions
    failed = predict()

    if failed:
        continue

    # evaluate
    out = subprocess.run(
        ["python", "eval_predictions.py", "--split", "dev"],
        capture_output=True,
        text=True,
    )
    print(out)
    f1 = round(float(out.stdout.strip("\n")), 5)
    row_lines.append(
        {
            "c": comb[0],
            "p": comb[1],
            "o": comb[2],
            "w": comb[3],
            "t": comb[4],
            "n_dim": comb[5],
            "f1": f1,
        },
    )
    print(comb)
    print(f1)

pd.DataFrame(row_lines).to_csv("./data/results_grid_search.csv", index=False)
