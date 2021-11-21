
from enum import IntEnum, auto
from musOracle import IBM, IRM, MWF
import musdb

class Oracle(IntEnum):
    IBM = auto()
    IRM = auto()
    MWF = auto()

_ORACLE_NAMES = ["IBM", "IRM", "MWF"]
_ORACLE_FUNCS = [IBM.IBM, IRM.IRM, MWF.MWF]

def get_name(oracle):
    return _ORACLE_NAMES[oracle] if oracle < len(_ORACLE_NAMES) else None

def get_id(oracle_name):
    for value,name in enumerate(_ORACLE_NAMES):
        if (name == oracle_name):
            return value
    return None

def predict_track(track, oracle):
    '''
    :param track: musdb's track instance.
    :type track: musdb.audio_classes.MultiTrack
    :param oracle: The oracle
    :type oracle: Oracle
    '''
    return _ORACLE_FUNCS[oracle](track)


def predict_db(db, oracle, estimates_dir):
    '''
    :param db: musdb's DB instance.
    :type db: musdb.DB
    :param oracle: The oracle
    :type oracle: Oracle
    '''
    for track in db:
        db.save_estimates(predict_track(track, oracle), track, estimates_dir)