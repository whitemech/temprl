import pickle
from abc import ABC

DEFAULT_DATA_DIR = "data"
DEFAULT_TRAINER_FILEPATH = DEFAULT_DATA_DIR + "/" + "trainer.pkl"

class Trainer(ABC):
    @staticmethod
    def load(datadir=DEFAULT_DATA_DIR):
        with open(datadir + "/trainer.pkl", "rb") as fin:
            return pickle.load(fin)

    @staticmethod
    def resume(datadir=DEFAULT_DATA_DIR, render: bool = False, verbosity:int=1):
        trainer = Trainer.load(datadir)
        return trainer.main(render=render, verbosity=verbosity)

    @staticmethod
    def eval(datadir=DEFAULT_DATA_DIR, render: bool = False, verbosity:int=1):
        trainer = Trainer.load(datadir)
        return trainer.main(eval=True, render=render, verbosity=verbosity)
