import argparse
import subprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_dim", type=int, required=False, default=200)
    args = parser.parse_args()

    n_dim = args.n_dim

    # create training data
    _ = subprocess.run(
        ["python", "prepare_train_data.py", "--n_dim", str(n_dim)],
    )

    # create testing data
    _ = subprocess.run(
        ["python", "prepare_dev_test_data.py", "--n_dim", str(n_dim)],
    )
