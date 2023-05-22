class BaseVisualization:
    def __init__(self, logger):
        self.logger = logger

    def log(self, outputs: dict, global_step: int):
        raise NotImplementedError