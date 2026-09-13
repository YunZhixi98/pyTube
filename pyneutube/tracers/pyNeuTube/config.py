# tracers/pyNeuTube/config.py
from dataclasses import dataclass
from math import floor
from enum import Enum, auto

# Global Defaults
@dataclass(frozen=True)
class Defaults:
    SEG_LENGTH: float = 11
    MIN_SEG_RADIUS: float = 0.5

    MIN_SEED_SCORE: float = 0.3
    MAX_SEED_RADIUS: float = 25
    MAX_CONF_RADIUS: float = 3

    TRACE_STEP: float = 0.5
    MAX_SEG_NUM: int = 5000
    MIN_SEG_SCORE: float = 0.3
    STOP_SEG_TRACE_SCORE: float = 0.3
    MIN_CHAIN_LENGTH: int = floor(SEG_LENGTH * 2.5)

    MIN_CHAIN_SCORE = 0.6
    CROSSOVER_TEST: bool = False

    # "grid": original C NeuTube-style 307 directions.
    # "hemisphere_uniform": 200 golden-angle upper-hemisphere axis samples.
    # "hemisphere_uniform_refine": 96 samples plus local unit-vector refinement.
    ORIENTATION_SEARCH_MODE: str = "hemisphere_uniform_refine"

    Max_Edge_Capacity: int = 1073741824 # 1G, maybe larger if necessary


class Optimization:
    MAX_ITER: int = 500

    DELTA_RADIUS: float = 0.1
    DELTA_THETA: float = 0.015
    DELTA_PSI: float = 0.015
    DELTA_SCALE: float = 0.05

    LINE_SEARCH_STOP_GRADIENT: float = 1e-1

class TraceStatus(Enum):
    NORMAL = 0
    HIT_MARK = auto()
    OUT_OF_BOUND = auto()
    VOXEL_TRACED = auto()
    LOW_SCORE = auto()
    SEG_TOO_THICK = auto()
    LOOP_FORMED = auto()
    SIGNAL_CHANGED = auto()
    RADIUS_CHANGED = auto()
    SEG_TOO_THIN = auto()


class TraceDirection(Enum):
    FORWARD = 0
    BACKWARD = auto()
    BOTH = auto()  # seed point
    # UNKNOWN = auto()  # inserted seg after tracing, like interpolated seg
    

class ConnectorType(Enum):    # status used for `mode` of ChainConnector
    NEUROCOMP_CONN_NONE = 0     # not connected
    NEUROCOMP_CONN_HL = auto()  # hook-loop mode
    NEUROCOMP_CONN_LINK = auto()    # link mode

#class GraphType(Enum):
#    GENERAL_GRAPH = 0
#    TREE_GRAPH = auto()
#    COMPLETE_GRAPH = auto()


