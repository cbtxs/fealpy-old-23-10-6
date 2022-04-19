import numpy as np

from fealpy.decorator import cartesian
from fealpy.mesh import MeshFactory as MF
from fealpy.geometry import DistDomain2d
from fealpy.mesh import DistMesh2d
from fealpy.geometry import dcircle,drectangle,ddiff,dmin
from fealpy.geometry import huniform
class SinCosData:
    """
    [0, 1]^2
    u(x, y) = (sin(pi*x)*cos(pi*y), -cos(pi*x)*sin(piy))
    p = 1/(y**2 + 1) - pi/4
    """
    def __init__(self):
        self.box = [0, 1, 0, 1]

    def domain(self):
        return self.box

    @cartesian
    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        cos = np.cos
        sin = np.sin
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = sin(pi*x)*cos(pi*y) 
        val[..., 1] = -cos(pi*x)*sin(pi*y) 
        return val

    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 1/(y**2 + 1) - pi/4 
        return val
    
    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = 2*pi**2*sin(pi*x)*cos(pi*y) + pi*sin(pi*x)*cos(pi*x)
        val[..., 1] = -2*y/(y**2 + 1)**2 - 2*pi**2*sin(pi*y)*cos(pi*x) + pi*sin(pi*y)*cos(pi*x) 
        return val


    @cartesian
    def dirichlet(self, p):
        return self.velocity(p)

class Poisuille:
    """
    [0, 1]^2
    u(x, y) = (4y(1-y), 0)
    p = 8(1-x)
    """
    def __init__(self):
        self.box = [0, 1, 0, 1]

    def domain(self):
        return self.box

    @cartesian
    def velocity(self, p):
        x = p[...,0]
        y = p[...,1]
        value = np.zeros(p.shape)
        value[...,0] = 4*y*(1-y)
        return value

    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = 8*(1-x) 
        return val
    
    @cartesian
    def source(self, p):
        val = np.zeros(p.shape, dtype=np.float)
        return val

    @cartesian
    def is_p_boundary(p):
        return (np.abs(p[..., 0]) < eps) | (np.abs(p[..., 0] - 1.0) < eps)
      
    @cartesian
    def is_wall_boundary(p):
        return (np.abs(p[..., 1]) < eps) | (np.abs(p[..., 1] - 1.0) < eps)

    @cartesian
    def dirichlet(self, p):
        return self.velocity(p)

class FlowPastCylinder:
    '''
    @brief 圆柱绕流
    '''
    def __init__(self, eps=1e-12, rho=1, mu=0.001):
        self.eps = eps
        self.rho = rho
        self.mu = mu

    def mesh(self,h): 
        points = np.array([[0.0, 0.0], [2.2, 0.0], [2.2, 0.41], [0.0, 0.41]],
                dtype=np.float64)
        facets = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int_)


        p, f = MF.circle_interval_mesh([0.2, 0.2], 0.05, 0.01) 

        points = np.append(points, p, axis=0)
        facets = np.append(facets, f+4, axis=0)

        fm = np.array([0, 1, 2, 3])

        smesh = MF.meshpy2d(points, facets, h, hole_points=[[0.2, 0.2]], facet_markers=fm, meshtype='tri')
        return smesh
    
    @cartesian
    def is_outflow_boundary(self,p):
        return np.abs(p[..., 0] - 2.2) < self.eps
    
    @cartesian
    def is_inflow_boundary(self,p):
        return np.abs(p[..., 0]) < self.eps
    
    @cartesian
    def is_circle_boundary(self,p):
        x = p[...,0]
        y = p[...,1]
        return (np.sqrt(x**2 + y**2) - 0.05) < self.eps
      
    @cartesian
    def is_wall_boundary(self,p):
        return (np.abs(p[..., 1] -0.41) < self.eps) | \
               (np.abs(p[..., 1] ) < self.eps)

    @cartesian
    def u_inflow_dirichlet(self, p):
        x = p[...,0]
        y = p[...,1]
        value = np.zeros(p.shape,dtype=np.float)
        value[...,0] = 1.5*4*y*(0.41-y)/(0.41**2)
        value[...,1] = 0
        return value
    
