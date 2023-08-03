import numpy as np
from fealpy.functionspace import ConformingScalarVESpace2d 
from fealpy.functionspace.conforming_vector_ve_space_2d import ConformingVectorVESpace2d
from fealpy.vem import ScaledMonomialSpaceMassIntegrator2d 
from fealpy.mesh import PolygonMesh
from fealpy.vem.conforming_vector_vem_h1_projector import ConformingVectorVEMH1Projector2d 
from fealpy.quadrature import GaussLobattoQuadrature
import ipdb
def test_assembly_cell_righthand_side(p):
    node = np.array([
            (0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (1.0, 0.0), (1.0, 1.0), (1.0, 2.0),
            (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)], dtype=np.float64)
    cell = np.array([0, 3, 4, 4, 1, 0, 1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5],
                dtype=np.int_)
    cellLocation = np.array([0, 3, 6, 10, 14, 18], dtype=np.int_)
    mesh = PolygonMesh(node, cell, cellLocation)

    
    a = mesh.entity_barycenter()
    nm = mesh.edge_normal()
    h = np.sqrt(mesh.cell_area())
    qf = GaussLobattoQuadrature(p+1)
    bcs, ws = qf.quadpts, qf.weights
    space =  ConformingVectorVESpace2d(mesh, p=p)
    m = ScaledMonomialSpaceMassIntegrator2d()
    M = m.assembly_cell_matrix(space.smspace) #(n_{k-1}, n_{k-1})
    projector = ConformingVectorVEMH1Projector2d(M)
    #ipdb.set_trace()
    B = projector.assembly_cell_right_hand_side(space) 
def test_assembly_cell_left_hand_side(p):
    node = np.array([
            (0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (1.0, 0.0), (1.0, 1.0), (1.0, 2.0),
            (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)], dtype=np.float64)
    cell = np.array([0, 3, 4, 4, 1, 0, 1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5],
                dtype=np.int_)
    cellLocation = np.array([0, 3, 6, 10, 14, 18], dtype=np.int_)
    mesh = PolygonMesh(node, cell, cellLocation)
    bc = mesh.entity_barycenter()
    h = np.sqrt(mesh.cell_area())

    space =  ConformingVectorVESpace2d(mesh, p=p)
    sspace = ConformingScalarVESpace2d(mesh, p=p)
    m = ScaledMonomialSpaceMassIntegrator2d()
    #ipdb.set_trace()
    M = m.assembly_cell_matrix(sspace.smspace) 
    projector = ConformingVectorVEMH1Projector2d(M)
    
    G = projector.assembly_cell_left_hand_side(space) 
def test_assembly_cell_matrix(p):
    node = np.array([
            (0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (1.0, 0.0), (1.0, 1.0), (1.0, 2.0),
            (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)], dtype=np.float64)
    cell = np.array([0, 3, 4, 4, 1, 0, 1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5],
                dtype=np.int_)
    cellLocation = np.array([0, 3, 6, 10, 14, 18], dtype=np.int_)
    mesh = PolygonMesh(node, cell, cellLocation)

    bc = mesh.entity_barycenter()
    h = np.sqrt(mesh.cell_area())

    space =  ConformingVectorVESpace2d(mesh, p=p)
    sspace = ConformingScalarVESpace2d(mesh, p=p)
    m = ScaledMonomialSpaceMassIntegrator2d()
    M = m.assembly_cell_matrix(sspace.smspace) 
    projector = ConformingVectorVEMH1Projector2d(M)
    PI1 = projector.assembly_cell_matrix(space)    

if __name__ == "__main__":
    test_assembly_cell_righthand_side(3)
    test_assembly_cell_left_hand_side(3)
    test_assembly_cell_matrix(3)
