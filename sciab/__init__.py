from enum import Enum

class SimStatus(Enum):
    """Simulation status"""
    SIM_TERMINATED = 0
    SIM_INFEASIBLE = 1
    SIM_UNSAFE = 2
    SIM_MAX_ITER_REACHED = 3
