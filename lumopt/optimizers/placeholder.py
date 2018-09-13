
class placeholder_optimizer(object):

    done=False
    self_managing=False

    def __init__(self,max_iter):
        self.max_iter=max_iter

    def update(self):
        pass