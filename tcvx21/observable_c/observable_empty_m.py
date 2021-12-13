from .observable_m import Observable


class EmptyObservable(Observable):
    """
    An observable containing no data
    """

    def __init__(self):
        pass

    def plot(self, *args, **kwargs):
        pass

    @property
    def is_empty(self):
        return True
