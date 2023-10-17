def get_dataset_name(path_to_dataset: str):
    return path_to_dataset.split("/")[-1].split(".")[0]
