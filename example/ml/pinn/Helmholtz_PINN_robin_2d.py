import time

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from torch.optim import Adam
from matplotlib import pyplot as plt

from fealpy.mesh import TriangleMesh
from fealpy.ml.grad import gradient
from fealpy.ml.modules import Solution
from fealpy.ml.sampler import BoxBoundarySampler, ISampler


# 超参数
num_of_point_pde = 100
num_of_point_bc = 50
lr = 0.01
iteration = 150
wavenum = float(1)
NN = 64

# 定义网络层结构
net_1 = nn.Sequential(
    nn.Linear(2, NN, dtype=torch.float64),
    nn.Tanh(),
    nn.Linear(NN, NN//2, dtype=torch.float64),
    nn.Tanh(),
    nn.Linear(NN//2, NN//4, dtype=torch.float64),
    nn.Tanh(),
    nn.Linear(NN//4, 1, dtype=torch.float64)
)
net_2 = nn.Sequential(
    nn.Linear(2, NN, dtype=torch.float64),
    nn.Tanh(),
    nn.Linear(NN, NN//2, dtype=torch.float64),
    nn.Tanh(),
    nn.Linear(NN//2, NN//4, dtype=torch.float64),
    nn.Tanh(),
    nn.Linear(NN//4, 1, dtype=torch.float64)
)

# 网络实例化
s_1= Solution(net_1)
s_2= Solution(net_2)

# 选择优化器和损失函数
optim_1 = Adam(s_1.parameters(), lr=lr, betas=(0.9, 0.99))
optim_2 = Adam(s_2.parameters(), lr=lr, betas=(0.9, 0.99))
mse_cost_func = nn.MSELoss(reduction='mean')

# 采样器
samplerpde_1 = ISampler(
    num_of_point_pde, [[-0.5, 0.5], [-0.5, 0.5]], requires_grad=True)
samplerpde_2 = ISampler(
    num_of_point_pde, [[-0.5, 0.5], [-0.5, 0.5]], requires_grad=True)
samplerbc_1 = BoxBoundarySampler(
    num_of_point_bc, [-0.5, -0.5], [0.5, 0.5], requires_grad=True)
samplerbc_2 = BoxBoundarySampler(
    num_of_point_bc, [-0.5, -0.5], [0.5, 0.5], requires_grad=True)

# 真解
def solution(p: torch.Tensor) -> torch.Tensor:

    k = torch.tensor(wavenum)
    x = p[..., 0:1]
    y = p[..., 1:2]
    r = torch.sqrt(x**2 + y**2)
    val = torch.zeros(x.shape, dtype=torch.complex128)
    val[:] = torch.cos(k*r)/k
    c = torch.complex(torch.cos(k), torch.sin(
        k))/torch.complex(torch.special.bessel_j0(k), torch.special.bessel_j1(k))/k
    val -= c*torch.special.bessel_j0(k*r)
    return val

def solution_numpy_real(p: NDArray):
    sol = solution(torch.tensor(p))
    ret = torch.real(sol)
    return ret.detach().numpy()

def solution_numpy_imag(p: NDArray):
    sol = solution(torch.tensor(p))
    ret = torch.imag(sol)
    return ret.detach().numpy()

# 定义pde
def pde(p: torch.Tensor) -> torch.Tensor:

    k = torch.tensor(wavenum)
    u = torch.complex(s_1(p), s_2(p))
    x = p[..., 0:1]
    y = p[..., 1:2]
    r = torch.sqrt(x**2 + y**2)

    f = torch.zeros(x.shape, dtype=torch.complex128)
    f[:] = torch.sin(k*r)/r

    u_x_real, u_y_real = gradient(u.real, p, create_graph=True, split=True)
    u_x_imag, u_y_imag = gradient(u.imag, p, create_graph=True, split=True)
    u_xx_real, _ = gradient(u_x_real, p, create_graph=True, split=True)
    u_xx_imag, _ = gradient(u_x_imag, p, create_graph=True, split=True)
    _, u_yy_real = gradient(u_y_real, p, create_graph=True, split=True)
    _, u_yy_imag = gradient(u_y_imag, p, create_graph=True, split=True)
    u_xx = torch.complex(u_xx_real, u_xx_imag)
    u_yy = torch.complex(u_yy_real, u_yy_imag)

    return u_xx + u_yy + u + f

# 定义边界条件
def grad(p: torch.Tensor):
    """
    x*(I*sin(k) + cos(k))*besselj(1, R*k)/(R*(besselj(0, k) + I*besselj(1, k))) - x*sin(R*k)/R
    y*(I*sin(k) + cos(k))*besselj(1, R*k)/(R*(besselj(0, k) + I*besselj(1, k))) - y*sin(R*k)/R
    """
    k = torch.tensor(wavenum)
    x = p[..., 0:1]
    y = p[..., 1:2]
    r = torch.sqrt(x**2 + y**2)

    val = torch.zeros(p.shape, dtype=torch.complex128)
    t0 = torch.sin(k*r)/r
    c = torch.complex(torch.cos(k), torch.sin(k)) / \
        torch.complex(torch.special.bessel_j0(k), torch.special.bessel_j1(k))
    t1 = c*torch.special.bessel_j1(k*r)/r
    t2 = t1 - t0
    val[..., 0:1] = t2*x
    val[..., 1:2] = t2*y
    return val

def bc(p: torch.Tensor) -> torch.Tensor:

    k = torch.tensor(wavenum)
    u = torch.complex(s_1(p), s_2(p))
    x = p[..., 0]
    y = p[..., 1]
    n = torch.zeros_like(p)
    n[x > torch.abs(y), 0] = 1.0
    n[y > torch.abs(x), 1] = 1.0
    n[x < -torch.abs(y), 0] = -1.0
    n[y < -torch.abs(x), 1] = -1.0

    grad_u_real = gradient(u.real, p, create_graph=True, split=False)
    grad_u_imag = gradient(u.imag, p, create_graph=True, split=False)
    grad_u = torch.complex(grad_u_real, grad_u_imag)

    kappa = torch.complex(torch.tensor(0.0), k)
    g = (grad(p)*n).sum(dim=-1, keepdim=True) + kappa * solution(p)

    return (grad_u*n).sum(dim=-1, keepdim=True) + kappa * u - g

# 构建网格和有限元空间
mesh = TriangleMesh.from_box([-0.5 ,0.5, -0.5, 0.5], nx = 320,ny = 320 )

# 训练过程
start_time = time.time()
Error_real = []
Error_imag = []

for epoch in range(iteration+1):

    optim_1.zero_grad()
    optim_2.zero_grad()

    spde_1 = samplerpde_1.run()
    sbc_1 = samplerbc_1.run()
    outpde_1 = pde(spde_1)
    outbc_1 = bc(sbc_1)

    spde_2 = samplerpde_2.run()
    sbc_2 = samplerbc_2.run()
    outpde_2 = pde(spde_2)
    outbc_2 = bc(sbc_2)

    outpde_real = torch.real(outpde_1)
    outpde_imag = torch.imag(outpde_2)
    mse_pde_real = mse_cost_func(outpde_real, torch.zeros_like(outpde_real))
    mse_pde_imag = mse_cost_func(outpde_imag, torch.zeros_like(outpde_imag))

    outbc_real = torch.real(outbc_1)
    outbc_imag = torch.imag(outbc_2)
    mse_bc_real = mse_cost_func(outbc_real, torch.zeros_like(outbc_real))
    mse_bc_imag = mse_cost_func(outbc_imag, torch.zeros_like(outbc_imag))

    loss = 0.5*(0.05 * mse_pde_real + 0.95 * mse_bc_real) +\
            0.5*(0.05* mse_pde_imag + 0.95 * mse_bc_imag)

    loss.backward(retain_graph=True)
    optim_1.step()
    optim_2.step()

    if epoch % 10 == 0:

        error_real = s_1.estimate_error(solution_numpy_real, mesh, coordtype='c')
        error_imag = s_2.estimate_error(solution_numpy_imag, mesh, coordtype='c')

        Error_real.append(error_real)
        Error_imag.append(error_imag)

        print(f"Epoch: {epoch}, Loss: {loss}")
        print(f"Error_real:{error_real}, Error_imag:{error_imag}")
        print('\n')

end_time = time.time()     # 记录结束时间
training_time = end_time - start_time   # 计算训练时间
print("训练时间为：", training_time, "秒")

y_real = range(1, 10*len(Error_real) +1,10)
y_imag = range(1, 10*len(Error_imag) +1,10)
plt.plot(y_real, Error_real)
plt.plot(y_imag, Error_imag)

bc_ = np.array([1/3, 1/3, 1/3])
ps = torch.tensor(mesh.bc_to_point(bc_), dtype=torch.float64)

u_real = torch.real(solution(ps)).detach().numpy()
u_imag = torch.imag(solution(ps)).detach().numpy()
up_real = s_1(ps).detach().numpy()
up_imag = s_2(ps).detach().numpy()

#可视化
fig, axes = plt.subplots(2, 2)
mesh.add_plot(axes[0, 0], cellcolor=u_real, linewidths=0, aspect=1)
mesh.add_plot(axes[0, 1], cellcolor=u_imag, linewidths=0, aspect=1)
mesh.add_plot(axes[1, 0], cellcolor=up_real, linewidths=0, aspect=1)
mesh.add_plot(axes[1, 1], cellcolor=up_imag, linewidths=0, aspect=1)

plt.show()

