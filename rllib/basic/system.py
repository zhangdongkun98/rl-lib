
from .data import Data


def set_trace(local):
    try:
        self = local.pop('self')
        __class__ = local.pop('__class__')
    except KeyError:
        pass
    local = Data(**local)
    import pdb
    import rlcompleter
    pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
    pdb.set_trace()



