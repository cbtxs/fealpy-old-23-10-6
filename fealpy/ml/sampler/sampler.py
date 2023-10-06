from warnings import warn
from typing import (
    Tuple, List, Dict, Any, Generator, Type, Optional, Literal
)
from math import log2
import torch
from torch import Tensor, float64, device
import numpy as np

from . import functional as F


class Sampler():
    """
    The base class for all types of samplers.
    """
    nd: int = 0
    _weight: Tensor
    def __init__(self, enable_weight=False,
                 dtype=float64, device: device=None,
                 requires_grad: bool=False, **kwargs) -> None:
        """
        @brief Initializes a Sampler instance.

        @param m: The number of samples to generate.
        @param dtype: Data type of samples. Defaults to `torch.float64`.
        @param device: device.
        @param requires_grad: A boolean indicating whether the samples should\
                              require gradient computation. Defaults to `False`.
        """
        self.enable_weight = enable_weight
        self.dtype = dtype
        self.device = device
        self.requires_grad = bool(requires_grad)
        self._weight = torch.tensor(torch.nan, dtype=dtype, device=device)

    def run(self, n: int) -> Tensor:
        """
        @brief Generates samples.

        @return: A tensor with shape (n, GD) containing the generated samples.
        """
        raise NotImplementedError

    def weight(self) -> Tensor:
        """
        @brief Get weights of the latest sample points. The weight of a sample is\
               equal to the reciprocal of the sampling density.

        @return: A tensor with shape (m, 1).
        """
        return self._weight

    def load(self, n: int, epoch: int=1) -> Generator[torch.Tensor, None, None]:
        """
        @brief Return a generator to call `sampler.run()`.

        @param epoch: Iteration number, defaults to 1.

        @return: Generator.
        """
        for _ in range(epoch):
            yield self.run(n)


class ConstantSampler(Sampler):
    """
    A sampler generating constants.
    """
    def __init__(self, value: Tensor, requires_grad: bool=False, **kwargs) -> None:
        """
        @brief Build a sampler generats constants.

        @param value: A constant tensor.
        @param requires_grad: bool.
        """
        assert value.ndim == 2
        super().__init__(dtype=value.dtype, device=value.device,
                         requires_grad=requires_grad, **kwargs)
        self.value = value
        self.nd = value.shape[-1]
        if self.enable_weight:
            self._weight[:] = torch.tensor(0.0, dtype=self.dtype, device=value.device)

    def run(self, n: int) -> Tensor:
        ret = self.value.repeat(n)
        ret.requires_grad = self.requires_grad
        return ret


class ISampler(Sampler):
    """
    A sampler that generates samples independently in each axis.
    """
    def __init__(self, ranges: Any, dtype=float64,
                 device: device=None, requires_grad: bool=False, **kwargs) -> None:
        """
        @brief Initializes an ISampler instance.

        @param ranges: An object that can be converted to a `numpy.ndarray`,\
                       representing the ranges in each sampling axis.\
                       For example, if sampling x in [0, 1] and y in [4, 5],\
                       use `ranges=[[0, 1], [4, 5]]`, or `ranges=[0, 1, 4, 5]`.
        @param dtype: Data type of samples. Defaults to `torch.float64`.
        @param requires_grad: A boolean indicating whether the samples should\
                              require gradient computation. Defaults to `False`.\
                              See `torch.autograd.grad`

        @throws ValueError: If `ranges` has an unexpected shape.
        """
        super().__init__(dtype=dtype, device=device, requires_grad=requires_grad,
                         **kwargs)
        if isinstance(ranges, Tensor):
            ranges_arr = ranges.detach().clone().to(device=device)
        else:
            ranges_arr = torch.tensor(ranges, dtype=dtype, device=device)

        if ranges_arr.ndim == 2:
            self.nd = ranges_arr.shape[0]
            self.lows = ranges_arr[:, 0].reshape(self.nd, )
            self.highs = ranges_arr[:, 1].reshape(self.nd, )
        elif ranges_arr.ndim == 1:
            self.nd, mod = divmod(ranges_arr.shape[0], 2)
            if mod != 0:
                raise ValueError("If `ranges` is 1-dimensional, its length is"
                                 f"expected to be even, but got {mod}.")
            self.lows = ranges_arr[::2].reshape(self.nd, )
            self.highs = ranges_arr[1::2].reshape(self.nd, )
        else:
            raise ValueError(f"Unexpected `ranges` shape {ranges_arr.shape}.")

        self.deltas = self.highs - self.lows

    def run(self, n: int) -> Tensor:
        """
        @brief Generates independent samples in each axis.

        @return: A tensor with shape (n, GD) containing the generated samples.
        """
        ret = torch.rand((n, self.nd), dtype=self.dtype, device=self.device)\
            * self.deltas + self.lows
        ret.requires_grad = self.requires_grad
        if self.enable_weight:
            self._weight[:] = 1/n
            self._weight = self._weight.broadcast_to(n, self.nd)
        return ret


class BoxBoundarySampler(Sampler):
    """Generate samples on the boundaries of a multidimensional rectangle."""
    def __init__(self, p1: List[float], p2: List[float],
                 dtype=float64, device: device=None,
                 requires_grad: bool=False, **kwargs) -> None:
        """
        @brief Generate samples on the boundaries of a multidimensional rectangle.

        @param p1, p2: Object that can be converted to `torch.Tensor`.\
                       Points at both ends of the diagonal.
        @param dtype: Data type of samples. Defaults to `torch.float64`.
        @param requires_grad: bool. Defaults to `False`. See `torch.autograd.grad`.
        """
        super().__init__(dtype=dtype, device=device, requires_grad=requires_grad,
                         **kwargs)
        t1, t2 = torch.tensor(p1), torch.tensor(p2)
        if len(t1.shape) != 1:
            raise ValueError
        if t1.shape != t2.shape:
            raise ValueError("p1 and p2 should be in a same shape.")
        self.nd = int(t1.shape[0])
        data = torch.vstack([t1, t2]).T

        self.subs: List[ISampler] = []

        for d in range(t1.shape[0]):
            range1, range2 = data.clone(), data.clone()
            range1[d, :] = data[d, 0]
            range2[d, :] = data[d, 1]
            self.subs.append(ISampler(ranges=range1, dtype=dtype,
                              device=device, requires_grad=requires_grad))
            self.subs.append(ISampler(ranges=range2, dtype=dtype,
                              device=device, requires_grad=requires_grad))

    def run(self, mb: int) -> Tensor:
        """
        @brief Generate samples on the boundaries of a multidimensional rectangle.

        @param mb: int. Number of samples in each boundary.

        @return: Tensor.
        """
        if self.enable_weight:
            b = len(self.subs)
            self._weight[:] = 1/mb/b
            self._weight = self._weight.broadcast_to(mb * b, self.nd)
        return torch.cat([s.run(mb) for s in self.subs], dim=0)


##################################################
### Mesh samplers
##################################################

EType = Literal['cell', 'face', 'edge', 'node']

class MeshSampler(Sampler):
    """
    Sample in the specified entity of a mesh.
    """

    DIRECTOR: Dict[Tuple[Optional[str], Optional[str]], Type['MeshSampler']] = {}

    def __new__(cls, mesh, etype: EType, index=np.s_[:],
                mode: Literal['random', 'linspace']='random',
                dtype=float64, device: device=None,
                requires_grad: bool=False):
        mesh_name = mesh.__class__.__name__
        ms_class = cls._get_sampler_class(mesh_name, etype)
        return object.__new__(ms_class)

    @classmethod
    def _assigned(cls, mesh_name: Optional[str], etype: Optional[str]='cell'):
        if (mesh_name, etype) in cls.DIRECTOR.keys():
            if mesh_name is None:
                mesh_name = "all types of mesh"
            if etype is None:
                etype = "entitie"
            raise KeyError(f"{etype}s in {mesh_name} has already assigned to "
                           "another mesh sampler.")
        cls.DIRECTOR[(mesh_name, etype)] = cls

    @classmethod
    def _get_sampler_class(cls, mesh_name: str, etype: EType):
        if etype not in {'cell', 'face', 'edge', 'node'}:
            raise ValueError(f"Invalid etity type name '{etype}'.")
        ms_class = cls.DIRECTOR.get((mesh_name, etype), None)
        if ms_class is None:
            ms_class = cls.DIRECTOR.get((mesh_name, None), None)
            if ms_class is None:
                ms_class = cls.DIRECTOR.get((None, etype), None)
                if ms_class is None:
                    raise NotImplementedError(f"Sampler for {mesh_name}'s {etype} "
                                              "has not been implemented.")
        return ms_class

    def __init__(self, mesh, etype: EType, index=np.s_[:],
                 mode: Literal['random', 'linspace']='random',
                 dtype=float64, device: device=None,
                 requires_grad: bool=False, **kwargs) -> None:
        """
        @brief Generate samples in the specified entities of a mesh.

        @param mesh: Mesh.
        @param etype: 'cell', 'face' or 'edge'. Type of entity to sample from.
        @param index: Index of entities to sample from.
        @param mode: 'random' or 'linspace'.
        @param dtype: Data type of samples. Defaults to `torch.float64`.
        @param requires_grad: bool. Defaults to `False`. See `torch.autograd.grad`.
        """
        self.etype = etype
        self.node = torch.tensor(mesh.entity('node'), dtype=dtype, device=device)
        self.nd = self.node.shape[-1]
        self.node = self.node.reshape(-1, self.nd)
        try:
            self.cell = torch.tensor(mesh.entity(etype, index=index), device=device)
        except TypeError:
            warn(f"{mesh.__class__.__name__}.entity() does not support the 'index' "
                 "parameter. The entity is sliced after returned.")
            self.cell = torch.tensor(mesh.entity(etype)[index, :], device=device)
        self.NVC: int = self.cell.shape[-1]
        self.mode = mode

        super().__init__(dtype=dtype, device=device, requires_grad=requires_grad,
                         **kwargs)

    # def _set_weight(self, mp: int) -> None:
    #     raw = self.mesh.entity_measure(etype=self.etype)
    #     raw /= mp * np.sum(raw, axis=0)
    #     if isinstance(raw, (float, int)):
    #         arr = torch.tensor([raw, ], dtype=self.dtype).broadcast_to(self.cell.shape[0], 1)
    #     elif isinstance(raw, np.ndarray):
    #         arr = torch.from_numpy(raw)[:, None]
    #     else:
    #         raise TypeError(f"Unsupported return from entity_measure method.")
    #     self._weight = arr.repeat(1, mp).reshape(-1, 1).to(device=self.device)

    def get_bcs(self, mp: int, n: int):
        """
        @brief Generate bcs according to the current mode.

        `mp` is the number of samples in 'random' mode, and is the order of\
        multiple indices in 'linspace' mode.
        """
        if self.mode == 'random':
            return F.random_weights(mp, n, dtype=self.dtype, device=self.device)
        elif self.mode == 'linspace':
            return F.linspace_weights(mp, n, dtype=self.dtype, device=self.device)
        else:
            raise ValueError(f"Invalid mode {self.mode}.")

    def cell_bc_to_point(self, bcs: Tensor) -> Tensor:
        """
        The optimized version of method `mesh.cell_bc_to_point()`
        to support faster sampling.
        """
        node = self.node
        cell = self.cell
        return torch.einsum('...j, ijk->...ik', bcs, node[cell])


class _PolytopeSampler(MeshSampler):
    """Sampler in all homogeneous polytope mesh cells, such as triangle mesh and\
        tetrahedron mesh."""
    def run(self, mp: int) -> Tensor:
        self.bcs = self.get_bcs(mp, self.NVC)
        return self.cell_bc_to_point(self.bcs).reshape((-1, self.nd))

_PolytopeSampler._assigned(None, 'edge')
_PolytopeSampler._assigned('IntervalMesh', 'cell')
_PolytopeSampler._assigned('TriangleMesh', None)
_PolytopeSampler._assigned('TetrahedronMesh', None)
_PolytopeSampler._assigned('QuadrangleMesh', 'face')
_PolytopeSampler._assigned('PolygonMesh', 'face')


class _QuadSampler(MeshSampler):
    """Sampler in a quadrangle mesh."""
    def run(self, mp: int) -> Tensor:
        bc_0 = self.get_bcs(mp, 2)
        bc_1 = self.get_bcs(mp, 2)
        if self.mode == 'linspace':
            self.bcs = F.multiply(bc_0, bc_1, mode='cross', order=[0, 2, 3, 1])
        else:
            self.bcs = F.multiply(bc_0, bc_1, mode='dot', order=[0, 2, 3, 1])
        return self.cell_bc_to_point(self.bcs).reshape((-1, self.nd))

_QuadSampler._assigned('QuadrangleMesh', 'cell')
_QuadSampler._assigned('HexahedronMesh', 'face')


class _UniformSampler(MeshSampler):
    """Sampler in a 2-d uniform mesh."""
    def run(self, mp: int) -> Tensor:
        ND = int(log2(self.NVC))
        bc_list = [self.get_bcs(mp, 2) for _ in range(ND)]
        if self.mode == 'linspace':
            self.bcs = F.multiply(*bc_list, mode='cross')
        else:
            self.bcs = F.multiply(*bc_list, mode='dot')
        return self.cell_bc_to_point(self.bcs).reshape((-1, self.nd))

_UniformSampler._assigned('UniformMesh1d', None)
_UniformSampler._assigned('UniformMesh2d', None)
_UniformSampler._assigned('UniformMesh3d', None)
