import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh, QuadrangleMesh, PolygonMesh
from fealpy.mesh.halfedge_mesh import HalfEdgeMesh2d 
from fealpy.mesh import HalfEdgeMesh2d as HM

mesh = QuadrangleMesh.from_box([0, 2, 0, 1], nx=2, ny=1)
#mesh = TriangleMesh.from_one_triangle()
hmesh = HalfEdgeMesh2d.from_mesh(mesh)
hmesh.refine_poly(np.array([1, 0], dtype=np.bool_))
hmesh.coarsen_poly(np.array([0, 0, 1, 1, 1], dtype=np.bool_))
#hmesh.refine_poly(np.array([1, 0], dtype=np.bool_))
#NC = hmesh.ds.number_of_cells()
#mark = np.ones(NC, dtype=np.bool_)
#mark[[4, 5, 6, 19]] = False
#hmesh.coarsen_poly(mark)

#hmesh.print()

#fig = plt.figure()
#axes = fig.gca()
#hmesh.add_plot(axes, aspect=0.5)
#hmesh.find_node(axes, showindex=True)
#hmesh.find_cell(axes, showindex=True)
#hmesh.add_halfedge_plot(axes, showindex=True)
#plt.show()

def animation_plot(plot=True):
    cell = np.array([[0,1,2,3],[1,4,5,2]],dtype = np.int_)
    node = np.array([[0,0],[1,0],[1,1],[0,1],[2,0],[2,1]], dtype = np.float_)
    mesh = QuadrangleMesh(node, cell)
    mesh = HalfEdgeMesh2d.from_mesh(mesh)
    mesh.init_level_info()
    NE = mesh.ds.NE

    r = 0.5
    h = 5e-3
    k=0
    N = 10
    fig = plt.figure()
    axes = fig.gca()
    plt.ion()
    for i in range(N):
        c = np.array([i*(2/N), 0.8])
        k=0
        while True:
            halfedge = mesh.ds.halfedge
            pre = halfedge[:, 3]
            node = mesh.entity('node')
            flag = np.linalg.norm(node-c, axis=1)<r
            flag1 = flag[halfedge[:, 0]].astype(int)
            flag2 = flag[halfedge[pre, 0]].astype(int)
            isMarkedHEdge = flag1+flag2==1
            NC = mesh.number_of_cells()
            isMarkedCell = np.zeros(NC, dtype=np.bool_)
            isMarkedCell[halfedge[isMarkedHEdge, 1]] = True
            isMarkedCell = isMarkedCell & (mesh.cell_area()>h**2)
            if (~isMarkedCell).all():
                break
            mesh.refine_poly(isMarkedCell)
            k+=1
            print('加密',k,i,'次***************************')

            plt.cla()
            mesh.add_plot(axes, linewidths = 0.4, aspect=0.5)
            #mesh.find_cell(axes, showindex=True)
            plt.pause(0.01)

        k=0
        while k<10:
            halfedge = mesh.ds.halfedge
            pre = halfedge[:, 3]
            node = mesh.entity('node')
            flag = np.linalg.norm(node-c, axis=1)<r
            flag1 = flag[halfedge[:, 0]].astype(int)
            flag2 = flag[halfedge[pre, 0]].astype(int)
            isMarkedHEdge = flag1+flag2==1
            NC = mesh.number_of_cells()
            isMarkedCell = np.zeros(NC, dtype=np.bool_)
            isMarkedCell[halfedge[isMarkedHEdge, 1]] = True
            isMarkedCell = ~isMarkedCell & (mesh.cell_area()<0.5)

            mesh.coarsen_poly(isMarkedCell)
            if (~isMarkedCell).all():
                break
            k+=1
            print('循环',k,'次***************************')
            plt.cla()
            mesh.add_plot(axes, linewidths = 0.4, aspect=0.5)
            plt.pause(0.01)
    plt.ioff()
    plt.show()

def circle_plot(plot=True):
    cell = np.array([[0,1,2,3],[1,4,5,2]],dtype = np.int_)
    node = np.array([[0,0],[1,0],[1,1],[0,1],[2,0],[2,1]], dtype = np.float_)
    mesh = QuadrangleMesh(node, cell)
    mesh = HalfEdgeMesh2d.from_mesh(mesh)
    mesh.init_level_info()
    NE = mesh.ds.NE

    r = 0.5
    h = 1e-3
    k=0
    N = 10
    fig = plt.figure()
    axes = fig.gca()
    c = np.array([2*(2/N), 0.8])
    k=0
    while True:
        halfedge = mesh.ds.halfedge
        pre = halfedge[:, 3]
        node = mesh.entity('node')
        flag = np.linalg.norm(node-c, axis=1)<r
        flag1 = flag[halfedge[:, 0]].astype(int)
        flag2 = flag[halfedge[pre, 0]].astype(int)
        isMarkedHEdge = flag1+flag2==1
        NC = mesh.number_of_cells()
        isMarkedCell = np.zeros(NC, dtype=np.bool_)
        isMarkedCell[halfedge[isMarkedHEdge, 1]] = True
        isMarkedCell = isMarkedCell & (mesh.cell_area()>h**2)
        if (~isMarkedCell).all():
            break
        mesh.refine_poly(isMarkedCell)
        k+=1

        mesh.add_plot(axes, linewidths = 0.4, aspect=0.5)
        #mesh.find_cell(axes, showindex=True)
        plt.pause(0.001)

    plt.show()


#animation_plot()
circle_plot()





