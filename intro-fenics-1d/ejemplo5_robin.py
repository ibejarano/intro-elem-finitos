#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Actividad 2: Ecuacion de Poisson 1D con condicion de Robin y Neumann, hecho a mano 

Deformacion de una viga fija en sus extremos con carga distruibuida.

  -E*A*d2u/dx2 = g*rho   0 < x < l0
  E*A*u'(0) = K0*(u(0) - G0)
  E*A*u'(l0) = MG 
    
"""

from __future__ import print_function
import numpy as np #importo numpy y lo denomino np
import matplotlib.pyplot as plt
# import yaml

#Puntos de x0 a xNX
NX = 12 #numero de intervalos
NODOS = NX+1 #cantidad de NODOS

# Material
A = 0.01*0.01
E = 210.0e9
L0 = 1
ALPHA = E*A
VOL = A*L0
RHO = 7850
K0 = E/1000
G0 = 0
MG = 2.0
G = 9.81

uh = np.zeros((NX+1,1))
h = 1./(NX)

Apre = (2.*ALPHA/h)*np.eye(NODOS)
Apre[0][0] = ALPHA/h
Apre[-1][-1] = ALPHA/h
R = np.zeros((NODOS,NODOS))
R[0][0] = K0
R[-1][-1] = 0 # Ya valia 0 pero queria remarcarlo

rows, cols = np.indices((NODOS,NODOS))
row_vals = np.diag(rows, k=-1)
col_vals = np.diag(cols, k=-1)
z1 = np.zeros((NODOS,NODOS))
z1[row_vals, col_vals]=-ALPHA/h
row_vals = np.diag(rows, k=1)
col_vals = np.diag(cols, k=1)
z2 = np.zeros((NODOS,NODOS))
z2[row_vals, col_vals]=-ALPHA/h

A_mat = Apre+z1+z2 #Matriz de rigidez
f = RHO*VOL*G/L0
b = f*h*np.ones((NODOS,1))#vector
b[0] = b[0]/2
b[-1] = b[-1]/2
r = np.zeros((NODOS,1))
r[0] = G0*K0 # Condicion de Robin
r[-1] = MG # Condicion de Neumann

#Calculo la solución
uh = np.linalg.solve(A_mat+R, b+r)

fig, axs = plt.subplots(1,1)

xu = np.linspace(0, 1.0, NODOS,endpoint = True)
ut = RHO*G*(1.0*xu-xu*xu/2.0)/E
axs.plot(xu, ut, "r", label="Solución analítica con m=0")
axs.hlines((MG*1.0/(ALPHA))+ut[-1],0,1.0,linestyles="dashed", label="Solución analítica")

axs.plot(xu,uh,'rs',markersize=10, label="Solución FEM")

# data = dict(
#     A = 0.01*0.01,
#     E = 210.0e9,
#     L0 = 1,
#     ALPHA = E*A,
#     RHO = 7850,
#     K0 = K0,
#     G0 = 0,
#     MG = MG,
#     G = 9.81,
#     result = uh.tolist()
# )

# with open('a_mano_robin.yml', 'w') as outfile:
#     yaml.dump(data, outfile, default_flow_style=False)

plt.legend()
axs.set_xlabel("x: Posición de la barra [m]")
axs.set_ylabel("Desplazamientos [m]")
plt.show()
