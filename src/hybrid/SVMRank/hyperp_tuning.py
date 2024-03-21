import subprocess
import itertools
from tqdm import tqdm
import pandas as pd


def train_model(c, p, o, w, t, n_dim):
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
