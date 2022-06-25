from fenics import *
import matplotlib.pyplot as plt
import numpy as np
# import yaml
# Material y dimensiones
A = 0.01*0.01
E = 210.0e9
L0 = 1
ALPHA = E*A
RHO = 7850
K0 = E/1000
G0 = 0
MG = 2.0
G = 9.81

# FENICS CODE
## Boundary conditions
class TopBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        return on_boundary and near(x[0], 0.0, tol)

class BottomBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        l = 1.0
        return on_boundary and near(x[0], l, tol)

nx = 12
MINX = 0.0
MAXX = L0
mesh = IntervalMesh(nx, MINX, MAXX)
V0 = FunctionSpace(mesh, "CG", 1)
marcador_borde = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
b_top = TopBoundary()
b_top.mark(marcador_borde, 10)
b_bottom = BottomBoundary()
b_bottom.mark(marcador_borde, 20)
ds = Measure("ds", domain=mesh, subdomain_data=marcador_borde)

## Problem definition
u = TrialFunction(V0)
v = TestFunction(V0)
f = Constant(RHO*A*G)

a = ALPHA*dot( grad(u), grad(v) )*dx + K0*u*v*ds(10)
L = f*v*dx + K0*G0*v*ds(10) + MG*v*ds(20)
u = Function(V0)

# Solution
solve(a == L, u)

# Plotting
fig, axs = plt.subplots(1,1)

uh = u.compute_vertex_values(mesh)
xu = np.linspace(0.0, 1.0, len(uh), endpoint=True)
ut = RHO*G*(1.0*xu-xu*xu/2.0)/E
axs.plot(xu, uh, "r")
axs.hlines((MG*1.0/(ALPHA))+ut[-1],0,1.0,linestyles="dashed")

axs.plot(xu, ut, "b--")

data = dict(
    A = 0.01*0.01,
    E = 210.0e9,
    L0 = 1,
    ALPHA = E*A,
    RHO = 7850,
    K0 = K0,
    G0 = 0,
    MG = MG,
    G = 9.81,
    result = uh.tolist()
)

# with open('fenics_results_robin.yml', 'w') as outfile:
#     yaml.dump(data, outfile, default_flow_style=False)

plt.show()
