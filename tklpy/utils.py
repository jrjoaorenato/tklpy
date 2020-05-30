#calculate kernel with sklearn
from enum import Enum

class KernelEnum(Enum):
    rbf = 0
    linear = 1
    lap = 2
