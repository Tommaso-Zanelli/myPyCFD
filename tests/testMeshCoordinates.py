#	MIT License
#
#	Copyright (c) 2023 Tommaso-Zanelli
#
#	Permission is hereby granted, free of charge, to any person obtaining a copy
#	of this software and associated documentation files (the "Software"), to deal
#	in the Software without restriction, including without limitation the rights
#	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#	copies of the Software, and to permit persons to whom the Software is
#	furnished to do so, subject to the following conditions:
#
#	The above copyright notice and this permission notice shall be included in all
#	copies or substantial portions of the Software.
#
#	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#	SOFTWARE.
#
#
# --------------------------------------------------------------------------- #
#!/usr/bin/env python3
# Modules
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sys
sys.path.append('../')
from mesh.cartesian_mesh import cartesian_mesh_t
from tools.combination_index import combination_index

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Parameters
verbose = False

# 2D mesh size
Nx_2D = 4 #4096
Ny_2D = 4 #4096

#3D mesh size
Nx_3D = 4 #256
Ny_3D = 4 #256
Nz_3D = 4 #256

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Test 2D mesh coordinates
test2Dmesh = cartesian_mesh_t((0, 1, 0, 1), (Nx_2D, Ny_2D))

fig2D = plt.figure(1)
for i in range(test2Dmesh.num_edges[0][0]):
    plt.plot(np.ones((test2Dmesh.num_edges[0][1])) * test2Dmesh.cell_faces[0][i], test2Dmesh.cell_faces[1][:], '-', color=(0.625, 0.625, 0.625))
for i in range(test2Dmesh.num_edges[0][1]):
    plt.plot(test2Dmesh.cell_faces[0][:], np.ones((test2Dmesh.num_edges[0][0])) * test2Dmesh.cell_faces[1][i], '-', color=(0.625, 0.625, 0.625))
plt.plot(test2Dmesh.cell_centre_array[0], test2Dmesh.cell_centre_array[1], 'bo')
plt.plot(test2Dmesh.cell_face_arrays[0][0], test2Dmesh.cell_face_arrays[0][1], 'ro')
plt.plot(test2Dmesh.cell_face_arrays[1][0], test2Dmesh.cell_face_arrays[1][1], 'go')
plt.plot(test2Dmesh.cell_edge_arrays[0][0], test2Dmesh.cell_edge_arrays[0][1], 'ko')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D mesh')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Test 3D mesh coordinates
print("\n\n")
test3Dmesh = cartesian_mesh_t((0, 1, 0, 1, 0, 1), (Nx_3D, Ny_3D, Nz_3D))


fig3D = plt.figure(2)
plt3D = plt.axes(projection='3d')
fig2D = plt.figure(1)
for i in range(test3Dmesh.num_corners[0][0]):
    for j in range(test3Dmesh.num_corners[0][1]):
        plt3D.plot3D(np.ones((test3Dmesh.num_corners[0][2])) * test3Dmesh.cell_faces[0][i], \
                     np.ones((test3Dmesh.num_corners[0][2])) * test3Dmesh.cell_faces[1][j], \
                     test3Dmesh.cell_faces[2][:], \
                     '-', color=(0.625, 0.625, 0.625))
for i in range(test3Dmesh.num_corners[0][0]):
    for j in range(test3Dmesh.num_corners[0][2]):
        plt3D.plot3D(np.ones((test3Dmesh.num_corners[0][1])) * test3Dmesh.cell_faces[0][i], \
                     test3Dmesh.cell_faces[1][:], \
                     np.ones((test3Dmesh.num_corners[0][1])) * test3Dmesh.cell_faces[2][j], \
                     '-', color=(0.625, 0.625, 0.625))
for i in range(test3Dmesh.num_corners[0][1]):
    for j in range(test3Dmesh.num_corners[0][2]):
        plt3D.plot3D(test3Dmesh.cell_faces[0][:], \
                     np.ones((test3Dmesh.num_corners[0][0])) * test3Dmesh.cell_faces[1][i], \
                     np.ones((test3Dmesh.num_corners[0][0])) * test3Dmesh.cell_faces[2][j], \
                     '-', color=(0.625, 0.625, 0.625))
plt3D.plot3D(test3Dmesh.cell_centre_array[0], test3Dmesh.cell_centre_array[1], \
             test3Dmesh.cell_centre_array[2], 'bo')
plt3D.plot3D(test3Dmesh.cell_face_arrays[0][0], test3Dmesh.cell_face_arrays[0][1], \
             test3Dmesh.cell_face_arrays[0][2], 'ro')
plt3D.plot3D(test3Dmesh.cell_face_arrays[1][0], test3Dmesh.cell_face_arrays[1][1], \
             test3Dmesh.cell_face_arrays[1][2], 'go')
plt3D.plot3D(test3Dmesh.cell_face_arrays[2][0], test3Dmesh.cell_face_arrays[2][1], \
             test3Dmesh.cell_face_arrays[2][2], 'yo')
plt3D.plot3D(test3Dmesh.cell_edge_arrays[0][0], test3Dmesh.cell_edge_arrays[0][1], \
             test3Dmesh.cell_edge_arrays[0][2], 'ko')
plt3D.plot3D(test3Dmesh.cell_edge_arrays[1][0], test3Dmesh.cell_edge_arrays[1][1], \
             test3Dmesh.cell_edge_arrays[1][2], 'o', color=(0.5, 0.5, 0.5))
plt3D.plot3D(test3Dmesh.cell_edge_arrays[2][0], test3Dmesh.cell_edge_arrays[2][1], \
             test3Dmesh.cell_edge_arrays[2][2], 'o', color=(0.75, 0.75, 0.75))
plt3D.plot3D(test3Dmesh.cell_corner_arrays[0][0], test3Dmesh.cell_corner_arrays[0][1], \
             test3Dmesh.cell_corner_arrays[0][2], 'mo')
plt3D.set_xlabel('X')
plt3D.set_ylabel('Y')
plt3D.set_zlabel('Z')
plt3D.set_title('3D mesh')
