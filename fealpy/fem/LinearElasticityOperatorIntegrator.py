import numpy as np

class LinearElasticityOperatorIntegrator:
    def __init__(self, lam, mu, q=3):
        self.lam = lam
        self.mu = mu
        self.q = q

    def assembly_cell_matrix(self, space, index=np.s_[:], cellmeasure=None, out=None):
        """
        construct the linear elasticity fem matrix
        """

        q = self.q
        lam = self.lam
        mu = self.mu

        GD = len(space) 
        mesh = space[0].mesh
        ldof = space[0].number_of_local_dofs()

        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)

        if GD == 2:
            idx = [(0, 0), (0, 1),  (1, 1)]
            imap = {(0, 0):0, (0, 1):1, (1, 1):2}
        elif GD == 3:
            idx = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
            imap = {(0, 0):0, (0, 1):1, (0, 2):2, (1, 1):3, (1, 2):4, (2, 2):5}

        A = []

        qf =  mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        grad = space[0].grad_basis(bcs, index=index) # (NQ, NC, ldof, GD)

        NC = len(cellmeasure)

        if out is None:
            K = np.zeros((NC, GD*ldof, GD*ldof), dtype=np.float64)
        else:
            assert out.shape == (NC, GD*ldof, GD*ldof)
            K = out

        A = [np.einsum('i, ijm, ijn, j->jmn', ws, grad[..., i], grad[..., j], cellmeasure, optimize=True) for i, j in idx]

        D = mu*np.sum(A)
        if space[0].doforder == 'nodes': # 先按节点顺序排 x 分量，再依次排 y、z 分量
            for i in range(GD):
                for j in range(i, GD):
                    if i == j:
                        K[:, i*ldof:(i+1)*ldof, i*ldof:(i+1)*ldof] += D 
                        K[:, i*ldof:(i+1)*ldof, i*ldof:(i+1)*ldof] += (mu + lam)*A[imap[(i, i)]]
                    else:
                        K[:, i*ldof:(i+1)*ldof, j*ldof:(j+1)*ldof] += lam*A[imap[(i, j)]] 
                        K[:, i*ldof:(i+1)*ldof, j*ldof:(j+1)*ldof] += mu*A[imap[(i, j)]].transpose(0, 2, 1)
                        K[:, j*ldof:(j+1)*ldof, i*ldof:(i+1)*ldof] += lam*A[imap[(i, j)]].tranpose(0, 2, 1)
                        K[:, j*ldof:(j+1)*ldof, i*ldof:(i+1)*ldof] += mu*A[imap[(i, j)]]
        elif space[0].doforder == 'vdims':
            for i in range(GD):
                for j in range(i, GD):
                    if i == j:
                        K[:, i::GD, i::GD] += D 
                        K[:, i::GD, i::GD] += (mu + lam)*A[imap[(i, i)]]
                    else:
                        K[:, i::GD, j::GD] += lam*A[imap[(i, j)]] 
                        K[:, i::GD, j::GD] += mu*A[imap[(i, j)]].transpose(0, 2, 1)
                        K[:, j::GD, i::GD] += lam*A[imap[(i, j)]].tranpose(0, 2, 1)
                        K[:, j::GD, i::GD] += mu*A[imap[(i, j)]]
        if out is None:
            return K


    def assembly_cell_matrix_fast(self, space, index=np.s_[:], cellmeasure=None):
        pass


    def assembly_cell_matrix_ref(self, space, index=np.s_[:], cellmeasure=None):
        pass
