from config import PROJECT_NAME
import wandb
from src.myutils import get_dataset_name


class WandbClient:
    def __init__(self, run_name=str):
        self.project_name = PROJECT_NAME
        self.run = wandb.init(project=self.project_name, name=run_name)

    def upload_model(self, pt_model):
        artifact = wandb.Artifact(name="first_attempt", type="model")
        artifact.add_file(local_path=pt_model, name="cool_model")
        self.run.log_artifact(artifact)

    def upload_dataset(self, dataset):
        artifact = wandb.Artifact(
            name=get_dataset_name(dataset),
            type="dataset",
        )
        artifact.add_file(local_path=dataset, name=get_dataset_name(dataset))
        self.run.log_artifact(artifact)

    def finish(self):
        wandb.finish()
