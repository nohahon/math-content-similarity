from config import PROJECT_NAME
import wandb


class WandbClient:
    def __init__(self, run_name=str):
        self.project_name = PROJECT_NAME
        wandb.init()
        wandb.init(project=self.project_name, name=run_name)

    def upload(self):
        pass

    def download(self):
        pass

    def log(self):
        pass
