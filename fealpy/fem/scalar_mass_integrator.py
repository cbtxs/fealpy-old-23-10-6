import numpy as np


class ScalarMassIntegrator:
    """
    @note (c u, v)
    """    

    def __init__(self, c=None, q=None):
        self.coef = c
        self.q = q

    def assembly_cell_matrix(self, space, index=np.s_[:], cellmeasure=None,
            out=None):
        """
        @note 没有参考单元的组装方式
        """

        q = self.q if self.q is not None else space.p+1
        coef = self.coef
        
        mesh = space.mesh
 
        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)

        NC = len(cellmeasure)
        ldof = space.number_of_local_dofs()  

        if out is None:
            M = np.zeros((NC, ldof, ldof), dtype=space.ftype)
        else:
            M = out

        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        phi0 = space.basis(bcs, index=index) # (NQ, NC, ldof)
        if coef is None:
            M += np.einsum('q, qci, qcj, c->cij', ws, phi0, phi0, cellmeasure, optimize=True)
        else:
            if callable(coef):
                if hasattr(coef, 'coordtype'):
                    if coef.coordtype == 'barycentric':
                        coef = coef(bcs)
                    elif coef.coordtype == 'cartesian':
                        ps = mesh.bc_to_point(bcs, index=index)
                        coef = coef(ps)
                else:
                    ps = mesh.bc_to_point(bcs, index=index)
                    coef = coef(ps)
            if np.isscalar(coef):
                M += coef*np.einsum('q, qci, qcj, c->cij', ws, phi0, phi0, cellmeasure, optimize=True)
            elif isinstance(coef, np.ndarray): 
                if coef.shape == (NC, ):
                    M += np.einsum('q, c, qci, qcj, c->cij', ws, coef, phi0, phi0, cellmeasure, optimize=True)
                else:
                    M += np.einsum('q, qc, qci, qcj, c->cij', ws, coef, phi0, phi0, cellmeasure, optimize=True)
            else:
                raise ValueError("coef is not correct!")

        if out is None:
            return M
        
    
    def assembly_cell_matrix_fast(self, space0, _, index=np.s_[:], cellmeasure=None):
        """
        @brief 基于无数值积分的组装方式
        """
        mesh = space0.mesh 
        assert mesh.meshtype in ['tri', 'tet']

    def assembly_cell_matrix_ref(self, space0, _, index=np.s_[:], cellmeasure=None):
        """
        @note 基于参考单元的矩阵组装
        """
        pass
