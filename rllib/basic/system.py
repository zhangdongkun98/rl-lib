
from .data import Data


def set_trace(local):
    try:
        self = local.pop('self')
        __class__ = local.pop('__class__')
    except KeyError:
        pass
    local = Data(**local)
    locals().update(local.to_dict())
    import pdb
    import rlcompleter
    pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
    pdb.set_trace()



def get_class_name(x):
    return str(type(x))[8:-2]


def get_method_name():
    import inspect
    f_name = inspect.getframeinfo(inspect.currentframe().f_back)[2]
    return f_name



def spin(secs=5.0, func=None, args=(), kwargs={}):
    import time
    while True:
        if func != None:
            func(*args, **kwargs)
        time.sleep(secs)
    return







##############################################
############ silent ##########################
##############################################


from contextlib import redirect_stdout
from io import StringIO 

class NullIO(StringIO):
    def write(self, txt):
        pass

def silent(fn):
    """Decorator to silence functions."""
    def silent_fn(*args, **kwargs):
        with redirect_stdout(NullIO()):
            return fn(*args, **kwargs)
    return silent_fn


import os, sys
class HiddenPrints:
    """
    with HiddenPrints():
        print("This wont print")
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


