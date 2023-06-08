import numpy as np
import mgis.behaviour as mgis_bv
from fealpy.csm.mfront import compile_mfront_file

file = compile_mfront_file('material/Elasticity.mfront')

lib = "./libElasticity.so"  # 用实际路径替换

# 定义应变张量
eto = np.zeros(6)
eto[1] = 1.0
eto[2] = 1.0
h = mgis_bv.Hypothesis.Tridimensional # 表示是三维

# 加载行为模型
b = mgis_bv.load(lib, "Elasticity", h)

# 设置材料属性
m = mgis_bv.MaterialDataManager(b, 2) # 2 表示要处理的材料数量
mgis_bv.setMaterialProperty(m.s1, "YoungModulus", 150e9) # 设置材料属性
mgis_bv.setMaterialProperty(m.s1, "PoissonRatio", 0.3)
mgis_bv.setExternalStateVariable(m.s1, "Temperature", 293.15) # 设置外部状态变量

# 初始化局部变量
mgis_bv.update(m) # 更新材料数据
#b.gradients[0] = eto
m.s1.gradients[0:] = eto
it = mgis_bv.IntegrationType.IntegrationWithTangentOperator
dt = 0
mgis_bv.integrate(m, it, dt, 0, m.n)

idx = mgis_bv.getVariableSize(b.thermodynamic_forces[0], h)
sig = m.s1.thermodynamic_forces

Dt = m.K
# 输出结果
print("Predicted Stress:")
print(sig)
print("Tangent Stiffness:")
print(Dt)

