import numpy as np
from fenics import *
import matplotlib.pyplot as plt

nx = 5
minx, maxx = 0.0, 1.0
mesh = IntervalMesh(nx, minx, maxx)
V0 = FunctionSpace(mesh, "CG", 1)

def borde_D(x, on_boundary):
    tol = 1.E-14
    return on_boundary and near(x[0], 1.0, tol)

def borde_I(x, on_boundary):
    tol = 1.E-14
    return on_boundary and near(x[0], 0.0, tol)

bc_der = DirichletBC(V0, Constant(0.0), borde_D)
bc_iz = DirichletBC(V0, Constant(0.0), borde_I)
bc = [bc_iz, bc_der]

u = TrialFunction(V0)
v = TestFunction(V0)
f = Constant(1.0)

a = dot( grad(u), grad(v) )*dx
L = f*v*dx

u = Function(V0)
solve( a == L, u, bc)

print("Tipo de variable:", type(u))

test_x = 0.134543

print(f"Solucion en {test_x}:", u(test_x))
uh = u.compute_vertex_values(mesh)
print("cantidad de celdas:", nx)
print("cantidad de vertices", len(uh))

fig, axs = plt.subplots(1,1)

xu = np.linspace(0, 1, len(uh), endpoint=True)
xe = np.arange(0, 1, 0.001)
ue = -0.5*xe*(xe-1.)
axs.plot(xu, uh, 'ro', markersize=10)
axs.plot(test_x, u(test_x), 'bo', markersize=10)
axs.plot(xe, ue, 'b')
plt.show()
