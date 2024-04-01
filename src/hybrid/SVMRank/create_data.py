import argparse
import subprocess


def create_data():
    """
    Creates training and testing data for SVMRank model.

    Args:
        --n_dim (int, optional): Number of dimensions for the data. Defaults to 200.

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_dim", type=int, required=False, default=200)
    args = parser.parse_args()

    n_dim = args.n_dim

    # create training data
    _ = subprocess.run(
        ["python", "./scripts/prepare_train_data.py", "--n_dim", str(n_dim)],
    )

    # create testing data
    _ = subprocess.run(
        [
            "python",
            "./scripts/prepare_dev_test_data.py",
            "--n_dim",
            str(n_dim),
        ],
    )


if __name__ == "__main__":
    create_data()
