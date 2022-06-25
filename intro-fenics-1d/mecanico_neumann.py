import numpy as np
from fenics import *
import matplotlib.pyplot as plt

# definicion de barra o material
l0 = 1 # 1 metro
A = 0.01*0.01
E = 210.0e9
rho = 7850
vol = A*l0
g = 9.81
Mg = 2.0 # 2kg approx 20 N
# Definicion de malla
nx = 12
minx = 0.0
maxx = l0 - minx
mesh = IntervalMesh(nx, minx, maxx)
V0 = FunctionSpace(mesh, "CG", 1)

def borde_Arr(x, on_boundary):
    tol = 1.E-14
    return on_boundary and near(x[0], 0.0, tol)

class borde_Aba(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-14
        l0 = 1
        return on_boundary and near(x[0], l0, tol)

marcador_borde = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
b_ab = borde_Aba()
b_ab.mark(marcador_borde, 20)

ds = Measure("ds", domain=mesh, subdomain_data=marcador_borde)
bc_ar = DirichletBC(V0, Constant(0.0), borde_Arr)
bc = [bc_ar]

u = TrialFunction(V0)
v = TestFunction(V0)
f = Constant(rho*vol*g/l0)

a = A*E*dot( grad(u), grad(v) )*dx
L = f*v*dx+(Mg)*v*ds(20)

u = Function(V0)
solve( a == L, u, bc)

uh = u.compute_vertex_values(mesh)
print("cantidad de celdas:", nx)
print("cantidad de vertices", len(uh))
print("Masa barra", rho*vol*g)
print("Solucion analitica", (Mg*l0/(A*E)))

fig, axs = plt.subplots(1,1)

xu = np.linspace(0, 1, len(uh), endpoint=True)
xt = np.linspace(0, 1, 200, endpoint=True)
ut = rho*g*(l0*xt-xt*xt/2.0)/E 

axs.plot(xu, uh, 'ro', markersize=5)
axs.plot(xt, ut, 'b')
axs.hlines((Mg*l0/(A*E))+ut[-1],0,l0,linestyles="dashed")
plt.title("Campo de desplazamientos u(x)")
plt.show()
