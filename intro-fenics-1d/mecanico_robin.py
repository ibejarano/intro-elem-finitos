from fenics import *
import matplotlib.pyplot as plt
import numpy as np
# Material y dimensiones
A = 0.01*0.01
E = 210.0e9
L = 1
alpha = E*A
rho = 7850
k0 = E/1000
g0 = 0
Mg = 2.0
g = 9.81


## Boundary conditions
class TopBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        return on_boundary and near(x[0], 0.0, tol)

class BottomBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        L = 1.0
        return on_boundary and near(x[0], L, tol)


# FENICS CODE
nx = 12
minx = 0.0
maxx = L

# FENICS CODE
nx = 12
minx = 0.0
maxx = L
mesh = IntervalMesh(nx, minx, maxx)
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
f = Constant(rho*A*g)

a = alpha*dot( grad(u), grad(v) )*dx + k0*u*v*ds(10)
L = f*v*dx + k0*g0*v*ds(10) + Mg*v*ds(20)
u = Function(V0)
solve(a == L, u)



fig, axs = plt.subplots(1,1)

uh = u.compute_vertex_values(mesh)
xu = np.linspace(0.0, 1.0, len(uh), endpoint=True)  
ut = rho*g*(1.0*xu-xu*xu/2.0)/E 
axs.plot(xu, uh, "r")
axs.hlines((Mg*1.0/(alpha))+ut[-1],0,1.0,linestyles="dashed")


# Plotting


axs.plot(xu, ut, "b--")




plt.show()