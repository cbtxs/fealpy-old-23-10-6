from typing import Union
from numpy.typing import NDArray
import numpy as np

from ..mesh_data_structure import MeshDataStructure


class Mesh():
    """
    @brief The base class for mesh.

    """
    ds: MeshDataStructure
    node: NDArray
    itype: np.dtype
    ftype: np.dtype

    ### General Interfaces ###

    def number_of_nodes(self) -> int:
        return self.ds.NN

    def number_of_edges(self) -> int:
        return self.ds.number_of_edges()

    def number_of_faces(self) -> int:
        return self.ds.number_of_faces()

    def number_of_cells(self) -> int:
        return self.ds.number_of_cells()

    def number_of_nodes_of_cells(self) -> int:
        """Number of nodes in a cell"""
        return self.ds.number_of_vertices_of_cells()

    def number_of_edges_of_cells(self) -> int:
        """Number of edges in a cell"""
        return self.ds.number_of_edges_of_cells()

    def number_of_faces_of_cells(self) -> int:
        """Number of faces in a cell"""
        return self.ds.number_of_faces_of_cells()

    number_of_vertices_of_cells = number_of_nodes_of_cells

    def geo_dimension(self) -> int:
        """
        @brief Get geometry dimension of the mesh.
        """
        return self.node.shape[-1]

    def top_dimension(self) -> int:
        """
        @brief Get topology dimension of the mesh.
        """
        return self.ds.TD

    @staticmethod
    def multi_index_matrix(p: int, etype: int) -> NDArray:
        """
        @brief 获取 p 次的多重指标矩阵

        @param[in] p 正整数

        @return multiIndex  ndarray with shape (ldof, TD+1)
        """
        if etype == 3:
            ldof = (p+1)*(p+2)*(p+3)//6
            idx = np.arange(1, ldof)
            idx0 = (3*idx + np.sqrt(81*idx*idx - 1/3)/3)**(1/3)
            idx0 = np.floor(idx0 + 1/idx0/3 - 1 + 1e-4) # a+b+c
            idx1 = idx - idx0*(idx0 + 1)*(idx0 + 2)/6
            idx2 = np.floor((-1 + np.sqrt(1 + 8*idx1))/2) # b+c
            multiIndex = np.zeros((ldof, 4), dtype=np.int_)
            multiIndex[1:, 3] = idx1 - idx2*(idx2 + 1)/2
            multiIndex[1:, 2] = idx2 - multiIndex[1:, 3]
            multiIndex[1:, 1] = idx0 - idx2
            multiIndex[:, 0] = p - np.sum(multiIndex[:, 1:], axis=1)
            return multiIndex
        elif etype == 2:
            ldof = (p+1)*(p+2)//2
            idx = np.arange(0, ldof)
            idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
            multiIndex = np.zeros((ldof, 3), dtype=np.int_)
            multiIndex[:,2] = idx - idx0*(idx0 + 1)/2
            multiIndex[:,1] = idx0 - multiIndex[:,2]
            multiIndex[:,0] = p - multiIndex[:, 1] - multiIndex[:, 2]
            return multiIndex
        elif etype == 1:
            ldof = p+1
            multiIndex = np.zeros((ldof, 2), dtype=np.int_)
            multiIndex[:, 0] = np.arange(p, -1, -1)
            multiIndex[:, 1] = p - multiIndex[:, 0]
            return multiIndex

    def _shape_function(self, bc: NDArray, p: int =1, mi: NDArray=None):
        """
        @brief

        @param[in] bc 
        """
        if p == 1:
            return bc
        TD = bc.shape[-1] - 1
        if mi is None:
            mi = self.multi_index_matrix(p, etype=TD)
        c = np.arange(1, p+1, dtype=np.int_)
        P = 1.0/np.multiply.accumulate(c)
        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=self.ftype)
        A[..., 1:, :] = p*bc[..., None, :] - t.reshape(-1, 1)
        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)
        idx = np.arange(TD+1)
        phi = np.prod(A[..., mi, idx], axis=-1)
        return phi

    def _grad_shape_function(self, bc: NDArray, p: int =1, mi: NDArray=None) -> NDArray:
        """
        @brief 计算形状为 (..., TD+1) 的重心坐标数组 bc 中, 每一个重心坐标处的 p 次 Lagrange 形函数值关于该重心坐标的梯度。
        """
        TD = bc.shape[-1] - 1
        if mi is None:
            mi = self.multi_index_matrix(p, etype=TD)
        ldof = mi.shape[0] # p 次 Lagrange 形函数的个数

        c = np.arange(1, p+1)
        P = 1.0/np.multiply.accumulate(c)

        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=bc.dtype)
        A[..., 1:, :] = p*bc[..., None, :] - t.reshape(-1, 1)

        FF = np.einsum('...jk, m->...kjm', A[..., 1:, :], np.ones(p))
        FF[..., range(p), range(p)] = p
        np.cumprod(FF, axis=-2, out=FF)
        F = np.zeros(shape, dtype=bc.dtype)
        F[..., 1:, :] = np.sum(np.tril(FF), axis=-1).swapaxes(-1, -2)
        F[..., 1:, :] *= P.reshape(-1, 1)

        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)

        Q = A[..., mi, range(TD+1)]
        M = F[..., mi, range(TD+1)]

        shape = bc.shape[:-1]+(ldof, TD+1)
        R = np.zeros(shape, dtype=bc.dtype)
        for i in range(TD+1):
            idx = list(range(TD+1))
            idx.remove(i)
            R[..., i] = M[..., i]*np.prod(Q[..., idx], axis=-1)
        return R # (..., ldof, TD+1)

    def _bernstein_shape_function(self, bc: NDArray, p: int=1, mi: NDArray=None):
        """
        @brief 
        """
        TD = bc.shape[1]-1
        if mi is None:
            mi = self.multi_index_matrix(p, TD)
        ldof = mi.shape[0]

        shape = bc.shape[:-1]+(p+1, TD+1)
        B = np.ones(shape, dtype=bc.dtype)
        B[..., 1:, :] = bc[..., None, :]
        B = np.cumprod(B, axis=1)

        P = np.arange(p+1)
        P[0] = 1
        P = np.cumprod(P)
        B /= P[:, None]

        # B : (NQ, p+1, TD+1) 
        # B[:, multiIndex, np.arange(TD+1).reshape(1, -1)]: (NQ, ldof, TD+1)
        phi = P[-1]*np.prod(B[:, mi, np.arange(TD+1).reshape(1, -1)], axis=-1)

        return phi

    def _grad_bernstein_shape_function(self, bc: NDArray, p: int=1, mi:
            NDArray=None):
        """
        @brief 
        """

        TD = bc.shape[1]-1
        if mi is None:
            mi = self.multi_index_matrix(p, TD)
        ldof = mi.shape[0]

        shape = bc.shape[:-1] + (p+1, TD+1)
        B = np.ones(shape, dtype=bc.dtype)
        B[..., 1:, :] = bc[..., None, :]
        B = np.cumprod(B, axis=1)

        P = np.arange(p+1)
        P[0] = 1
        P = np.cumprod(P)
        B /= P[:, None]

        F = np.zeros(B.shape, dtype=bc.dtype)
        F[:, 1:] = B[:, :-1]

        shape = bc.shape[:-1]+(ldof, TD+1)
        R = np.zeros(shape, dtype=bc.dtype)
        for i in range(TD+1):
            idx = list(range(TD+1))
            idx.remove(i)
            idx = np.array(idx, dtype=np.int_)
            R[..., i] = np.prod(B[..., mi[:, idx], idx.reshape(1, -1)],
                    axis=-1)*F[..., mi[:, i], [i]]

        return P[-1]*R

    def shape_function(self, bc, p=1) -> NDArray:
        """
        @brief The cell shape function.
        """
        raise NotImplementedError

    def grad_shape_function(self, bc, p=1, variables='x', index=np.s_[:]):
        """
        @brief The gradient of the cell shape function.
        """
        raise NotImplementedError

    def uniform_refine(self, n: int=1) -> None:
        """
        @brief Refine the whole mesh uniformly for `n` times.
        """
        raise NotImplementedError


    def integrator(self, k: int, etype: Union[int, str]):
        """
        @brief Get the integration formula on a mesh entity of different dimensions.
        """
        raise NotImplementedError

    def bc_to_point(self, bc: NDArray, index=np.s_[:]) -> NDArray:
        """
        @brief Convert barycenter coordinate points to cartesian coordinate points\
               on mesh entities.

        @param bc: Barycenter coordinate points array, with shape (NQ, NVC), where\
                   NVC is the number of nodes in each entity.
        @param etype: Specify the type of entities on which the coordinates be converted.
        @param index: Index to slice entities.

        @note: To get the correct result, the order of bc must match the order of nodes\
               in the entity.

        @return: Cartesian coordinate points array, with shape (NQ, GD).
        """
        node = self.entity('node')
        TD = bc.shape[-1] - 1
        entity = self.entity(TD, index=index)
        p = np.einsum('...j, ijk -> ...ik', bc, node[entity])
        return p

    def number_of_entities(self, etype, index=np.s_[:]):
        raise NotImplementedError

    def entity(self, etype: Union[int, str], index=np.s_[:]) -> NDArray:
        """
        @brief Get entities.

        @param etype: Type of entities. Accept dimension or name.
        @param index: Index for entities.

        @return: A tensor representing the entities in this mesh.
        """
        TD = self.top_dimension()
        GD = self.geo_dimension()
        if etype in {'cell', TD}:
            return self.ds.cell[index]
        elif etype in {'edge', 1}:
            return self.ds.edge[index]
        elif etype in {'node', 0}:
            return self.node.reshape(-1, self.geo_dimension())[index]
        elif etype in {'face', TD-1}: # Try 'face' in the last
            return self.ds.face[index]
        raise ValueError(f"Invalid etype '{etype}'.")

    def entity_barycenter(self, etype: Union[int, str], index=np.s_[:]) -> NDArray:
        """
        @brief Calculate barycenters of entities.
        """
        node = self.entity('node')
        TD = self.ds.TD
        if etype in {'cell', TD}:
            cell = self.ds.cell
            return np.sum(node[cell[index], :], axis=1) / cell.shape[1]
        elif etype in {'edge', 1}:
            edge = self.ds.edge
            return np.sum(node[edge[index], :], axis=1) / edge.shape[1]
        elif etype in {'node', 0}:
            return node[index]
        elif etype in {'face', TD-1}: # Try 'face' in the last
            face = self.ds.face
            return np.sum(node[face[index], :], axis=1) / face.shape[1]
        raise ValueError(f"Invalid entity type '{etype}'.")

    def entity_measure(self, etype: Union[int, str], index=np.s_[:]) -> NDArray:
        """
        @brief Calculate measurements of entities.
        """
        raise NotImplementedError


    def number_of_local_ipoints(self, p: int, iptype: Union[int, str]='cell') -> int:
        """
        @brief Return the number of p-order interpolation points in a single entity.
        """
        raise NotImplementedError

    def number_of_global_ipoints(self, p: int) -> int:
        """
        @brief Return the number of all p-order interpolation points.
        """
        raise NotImplementedError

    def interpolation_points(self, p: int) -> NDArray:
        """
        @brief Get all the p-order interpolation points in the mesh.
        """
        raise NotImplementedError

    def node_to_ipoint(self, p: int, index=np.s_[:]) -> NDArray:
        return np.arange(self.number_of_nodes())[index]

    def edge_to_ipoint(self, p: int, index=np.s_[:]) -> NDArray:
        """
        @brief 获取网格边与插值点的对应关系
        """
        if isinstance(index, slice) and index == slice(None):
            NE = self.number_of_edges()
            index = np.arange(NE)
        elif isinstance(index, np.ndarray) and (index.dtype == np.bool_):
            index, = np.nonzero(index)
            NE = len(index)
        elif isinstance(index, list) and (type(index[0]) is np.bool_):
            index, = np.nonzero(index)
            NE = len(index)
        else:
            NE = len(index)

        NN = self.number_of_nodes()

        edge = self.entity('edge', index=index)
        edge2ipoints = np.zeros((NE, p+1), dtype=self.itype)
        edge2ipoints[:, [0, -1]] = edge
        if p > 1:
            idx = NN + np.arange(p-1)
            edge2ipoints[:, 1:-1] =  (p-1)*index[:, None] + idx
        return edge2ipoints

    def edge_length(self, index=np.s_[:], node=None):
        """
        @brief
        """
        node = self.entity('node') if node is None else node
        edge = self.entity('edge', index=index)
        v = node[edge[:, 1]] - node[edge[:,0]]
        return np.linalg.norm(v, axis=1)

    def edge_tangent(self, index=np.s_[:], node=None):
        node = self.entity('node') if node is None else node
        edge = self.entity('edge', index)
        v = node[edge[:, 1], :] - node[edge[:, 0], :]
        return v

    def edge_unit_tangent(self, index=np.s_[:], node=None):
        node = self.entity('node') if node is None else node
        edge = self.entity('edge', index=index)
        v = node[edge[:, 1], :] - node[edge[:, 0], :]
        length = np.sqrt(np.square(v).sum(axis=1))
        return v/length.reshape(-1, 1)

    def error(self, u, v, q=3, power=2, celltype=False):
        """
        @brief Calculate the error between two functions.
        """
        GD = self.geo_dimension()

        qf = self.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        ps = self.bc_to_point(bcs)

        if callable(u):
            if not hasattr(u, 'coordtype'):
                u = u(ps)
            else:
                if u.coordtype == 'cartesian':
                    u = u(ps)
                elif u.coordtype == 'barycentric':
                    u = u(bcs)

        if callable(v):
            if not hasattr(v, 'coordtype'):
                v = v(ps)
            else:
                if v.coordtype == 'cartesian':
                    v = v(ps)
                elif v.coordtype == 'barycentric':
                    v = v(bcs)

        if u.shape[-1] == 1:
           u = u[..., 0]

        if v.shape[-1] == 1:
           v = v[..., 0]

        cm = self.entity_measure('cell')

        f = np.power(np.abs(u - v), power)
        if isinstance(f, (int, float)): # f为标量常函数
            e = f*cm
        elif isinstance(f, np.ndarray):
            if f.shape == (GD, ): # 常向量函数
                e = cm[:, None]*f
            elif f.shape == (GD, GD):
                e = cm[:, None, None]*f
            else:
                e = np.einsum('q, qc..., c->c...', ws, f, cm)

        if celltype is False:
            e = np.power(np.sum(e), 1/power)
        else:
            e = np.power(np.sum(e, axis=tuple(range(1, len(e.shape)))), 1/power)
        return e # float or (NC, )
