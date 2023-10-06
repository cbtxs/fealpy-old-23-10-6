
from .module import TensorMapping, Solution, ZeroMapping, Fixed, Extracted, Projected
from .function_space import FunctionSpace, Function
from .linear import Standardize, Distance, MultiLinear
from .boundary import BoxDBCSolution, BoxDBCSolution1d, BoxDBCSolution2d, BoxNBCSolution
from .pikf import KernelFunctionSpace
from .rfm import RandomFeatureSpace

from .activate import Sin, Cos, Tanh, Besselj0
from .pou import PoUA, PoUSin, PoUSpace, UniformPoUSpace
from .loss import ScaledMSELoss
