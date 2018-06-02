import pickle
from abc import ABC

DEFAULT_DATA_DIR = "data"
DEFAULT_TRAINER_FILEPATH = DEFAULT_DATA_DIR + "/" + "trainer.pkl"

class Trainer(ABC):
    @staticmethod
    def load(filepath=DEFAULT_TRAINER_FILEPATH):
        with open(filepath, "rb") as fin:
            return pickle.load(fin)

    @staticmethod
    def resume(filepath=DEFAULT_TRAINER_FILEPATH, render: bool = False, verbosity:int=1):
        trainer = Trainer.load(filepath)
        return trainer.main(render=render, verbosity=verbosity)

    @staticmethod
    def eval(filepath=DEFAULT_TRAINER_FILEPATH, render: bool = False, verbosity:int=1):
        trainer = Trainer.load(filepath)
        return trainer.main(eval=True, render=render, verbosity=verbosity)
