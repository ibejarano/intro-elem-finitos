from fenics import *
import matplotlib.pyplot as plt
import numpy as np
# Geo definition
width = 0.01
height = 0.1
A = width*height
# Material definition
h = 42 
k = 64
qg = 1e6
Tinf = 353.15 

# FENICS CODE
# 1-D
# Mesh definition
nx = 10
minx, maxx = 0.0, width
mesh = IntervalMesh(nx, minx, maxx)
V0 = FunctionSpace(mesh, "CG", 2)

# Boundary Conditions
class BordeSuperior(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        return on_boundary and near(x[0], 0.0, tol)

class BordeInferior(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        return on_boundary and near(x[0], width, tol)

marcador_borde = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
bc_ab = BordeInferior()
bc_ar = BordeSuperior()

bc_ab.mark(marcador_borde, 20)
bc_ar.mark(marcador_borde, 30)
ds = Measure("ds", domain=mesh, subdomain_data=marcador_borde)

# # Formulacion variacional
T = TrialFunction(V0)
v = TestFunction(V0)
f = Constant(qg)

a = k * dot( grad(T), grad(v) )*dx + h*T*v*ds(20) + h*T*v*ds(30)
L = f*v*dx + h*Tinf*v*ds(20) + h*Tinf*v*ds(30)

# # Solve
T = Function(V0)
solve(a == L, T)


# PLOT AND PRESENTATION OF RESULTS
Th = T.compute_vertex_values(mesh)
print("Cantidad de celdas:", nx)
print("Cantidad de vertices", len(Th))
print(Th)
fig, axs = plt.subplots(1,1)

xu = np.linspace(0, width, len(Th), endpoint=True)

axs.plot(xu, Th-273.15, "ro", markersize=5)
axs.set_xlabel("X (m)")
axs.set_ylabel("Temperatura (°C)")
plt.title("Temperatura vs x")
plt.show()