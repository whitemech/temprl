class Policy(object):
    def __init__(self, eval:bool):
        self.eval = eval

    def choose_action(self, values=(), optimal=False):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        pass

    def set_eval(self, eval: bool):
        self.eval = eval
