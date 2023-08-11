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
import time
import sys
import numpy as np
sys.path.append('../')
from mesh.cartesian_mesh import cartesian_mesh_t
from tools.combination_index import combination_index

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Parameters
verbose = False

# 2D mesh size
Nx_2D = 4096
Ny_2D = 4096

#3D mesh size
Nx_3D = 256
Ny_3D = 256
Nz_3D = 256

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Counter initializations
c_error = 0
tot_time = 0.0
tot_comp = 0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Test 2D mesh indexing
test2Dmesh = cartesian_mesh_t((0, 1, 0, 1), (Nx_2D, Ny_2D))
# Test cells
if (verbose): print("Testing 2D cell indexing:\n")
error = 0
t1 = time.process_time()
for i in range(test2Dmesh.tot_cells):
    index_tuple = test2Dmesh.local_index(i)
    check_i = test2Dmesh.global_index(index_tuple)
    error += abs(i - check_i)
    #print("\t", i, " -> ", index_tuple, " -> ", check_i)
t2 = time.process_time()
cpu_time = t2 - t1
if (verbose): print("\nTotal error is: ", error)
if (verbose): print("Time elapsed is: ", cpu_time)
c_error += error
tot_time += cpu_time
tot_comp += test2Dmesh.tot_cells
if (verbose): print("===============================================================================")
# Test faces
if (verbose): print("Testing 2D faces indexing along x direction:\n")
error = 0
t1 = time.process_time()
for i in range(test2Dmesh.tot_faces[0]):
    index_tuple = test2Dmesh.local_index(i, 1, 0)
    check_i = test2Dmesh.global_index(index_tuple, 1, 0)
    error += abs(i - check_i)
    #print("\t", i, " -> ", index_tuple, " -> ", check_i)
t2 = time.process_time()
cpu_time = t2 - t1
if (verbose): print("\nTotal error is: ", error)
if (verbose): print("Time elapsed is: ", cpu_time)
c_error += error
tot_time += cpu_time
tot_comp += test2Dmesh.tot_faces[0]
if (verbose): print("===============================================================================")
if (verbose): print("Testing 2D faces indexing along y direction:\n")
error = 0
t1 = time.process_time()
for i in range(test2Dmesh.tot_faces[1]):
    index_tuple = test2Dmesh.local_index(i, 1, 1)
    check_i = test2Dmesh.global_index(index_tuple, 1, 1)
    error += abs(i - check_i)
    #print("\t", i, " -> ", index_tuple, " -> ", check_i)
t2 = time.process_time()
cpu_time = t2 - t1
if (verbose): print("\nTotal error is: ", error)
if (verbose): print("Time elapsed is: ", cpu_time)
c_error += error
tot_time += cpu_time
tot_comp += test2Dmesh.tot_faces[1]
if (verbose): print("===============================================================================")
# Test edges
if (verbose): print("Testing 2D edge indexing:\n")
error = 0
t1 = time.process_time()
for i in range(test2Dmesh.tot_edges[0]):
    index_tuple = test2Dmesh.local_index(i, 2, 0)
    check_i = test2Dmesh.global_index(index_tuple, 2, 0)
    error += abs(i - check_i)
    #print("\t", i, " -> ", index_tuple, " -> ", check_i)
t2 = time.process_time()
cpu_time = t2 - t1
if (verbose): print("\nTotal error is: ", error)
if (verbose): print("Time elapsed is: ", cpu_time)
c_error += error
tot_time += cpu_time
tot_comp += test2Dmesh.tot_edges[0]
print("===============================================================================")

print("Total time elapsed for 2D grid: ", tot_time, "s")
print("Total number of function calls: ", tot_comp)
print("Time elapsed per function call: ", tot_time / tot_comp, "s")
tot_time = 0
tot_comp = 0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Test 3D mesh indexing
print("\n\n")
test3Dmesh = cartesian_mesh_t((0, 1, 0, 1, 0, 1), (Nx_3D, Ny_3D, Nz_3D))
# Test cells
if (verbose): print("Testing 3D cell indexing:\n")
error = 0
t1 = time.process_time()
for i in range(test3Dmesh.tot_cells):
    index_tuple = test3Dmesh.local_index(i)
    check_i = test3Dmesh.global_index(index_tuple)
    error += abs(i - check_i)
    #print("\t", i, " -> ", index_tuple, " -> ", check_i)
t2 = time.process_time()
cpu_time = t2 - t1
if (verbose): print("\nTotal error is: ", error)
if (verbose): print("Time elapsed is: ", cpu_time)
c_error += error
tot_time += cpu_time
tot_comp += test3Dmesh.tot_cells
if (verbose): print("===============================================================================")
# Test faces
if (verbose): print("Testing 3D faces indexing along x direction:\n")
error = 0
t1 = time.process_time()
for i in range(test3Dmesh.tot_faces[0]):
    index_tuple = test3Dmesh.local_index(i, 1, 0)
    check_i = test3Dmesh.global_index(index_tuple, 1, 0)
    error += abs(i - check_i)
    #print("\t", i, " -> ", index_tuple, " -> ", check_i)
t2 = time.process_time()
cpu_time = t2 - t1
if (verbose): print("\nTotal error is: ", error)
if (verbose): print("Time elapsed is: ", cpu_time)
c_error += error
tot_time += cpu_time
tot_comp += test3Dmesh.tot_faces[0]
if (verbose): print("===============================================================================")
if (verbose): print("Testing 3D faces indexing along y direction:\n")
error = 0
t1 = time.process_time()
for i in range(test3Dmesh.tot_faces[1]):
    index_tuple = test3Dmesh.local_index(i, 1, 1)
    check_i = test3Dmesh.global_index(index_tuple, 1, 1)
    error += abs(i - check_i)
    #print("\t", i, " -> ", index_tuple, " -> ", check_i)
t2 = time.process_time()
cpu_time = t2 - t1
if (verbose): print("\nTotal error is: ", error)
if (verbose): print("Time elapsed is: ", cpu_time)
c_error += error
tot_time += cpu_time
tot_comp += test3Dmesh.tot_faces[1]
if (verbose): print("===============================================================================")
if (verbose): print("Testing 3D faces indexing along z direction:\n")
error = 0
t1 = time.process_time()
for i in range(test3Dmesh.tot_faces[2]):
    index_tuple = test3Dmesh.local_index(i, 1, 2)
    check_i = test3Dmesh.global_index(index_tuple, 1, 2)
    error += abs(i - check_i)
    #print("\t", i, " -> ", index_tuple, " -> ", check_i)
t2 = time.process_time()
cpu_time = t2 - t1
if (verbose): print("\nTotal error is: ", error)
if (verbose): print("Time elapsed is: ", cpu_time)
c_error += error
tot_time += cpu_time
tot_comp += test3Dmesh.tot_faces[2]
if (verbose): print("===============================================================================")
# Test edges
dirs = ["x", "y", "z"]
ci = combination_index(3, 2, 0)
if (verbose): print("Testing 3D edge indexing along ", dirs[ci[0]], " and ", dirs[ci[1]], " directions\n")
error = 0
t1 = time.process_time()
for i in range(test3Dmesh.tot_edges[0]):
    index_tuple = test3Dmesh.local_index(i, 2, 0)
    check_i = test3Dmesh.global_index(index_tuple, 2, 0)
    error += abs(i - check_i)
    #print("\t", i, " -> ", index_tuple, " -> ", check_i)
t2 = time.process_time()
cpu_time = t2 - t1
if (verbose): print("\nTotal error is: ", error)
if (verbose): print("Time elapsed is: ", cpu_time)
c_error += error
tot_time += cpu_time
tot_comp += test3Dmesh.tot_edges[0]
if (verbose): print("===============================================================================")
# Test edges
dirs = ["x", "y", "z"]
ci = combination_index(3, 2, 1)
if (verbose): print("Testing 3D edge indexing along ", dirs[ci[0]], " and ", dirs[ci[1]], " directions\n")
error = 0
t1 = time.process_time()
for i in range(test3Dmesh.tot_edges[1]):
    index_tuple = test3Dmesh.local_index(i, 2, 1)
    check_i = test3Dmesh.global_index(index_tuple, 2, 1)
    error += abs(i - check_i)
    #print("\t", i, " -> ", index_tuple, " -> ", check_i)
t2 = time.process_time()
cpu_time = t2 - t1
if (verbose): print("\nTotal error is: ", error)
if (verbose): print("Time elapsed is: ", cpu_time)
c_error += error
tot_time += cpu_time
tot_comp += test3Dmesh.tot_edges[1]
if (verbose): print("===============================================================================")
# Test edges
dirs = ["x", "y", "z"]
ci = combination_index(3, 2, 2)
if (verbose): print("Testing 3D edge indexing along ", dirs[ci[0]], " and ", dirs[ci[1]], " directions\n")
error = 0
t1 = time.process_time()
for i in range(test3Dmesh.tot_edges[2]):
    index_tuple = test3Dmesh.local_index(i, 2, 2)
    check_i = test3Dmesh.global_index(index_tuple, 2, 2)
    error += abs(i - check_i)
    #print("\t", i, " -> ", index_tuple, " -> ", check_i)
t2 = time.process_time()
cpu_time = t2 - t1
if (verbose): print("\nTotal error is: ", error)
if (verbose): print("Time elapsed is: ", cpu_time)
c_error += error
tot_time += cpu_time
tot_comp += test3Dmesh.tot_edges[2]
if (verbose): print("===============================================================================")
# Test corners
if (verbose): print("Testing 3D corner indexing\n")
error = 0
t1 = time.process_time()
for i in range(test3Dmesh.tot_corners[0]):
    index_tuple = test3Dmesh.local_index(i, 3, 0)
    check_i = test3Dmesh.global_index(index_tuple, 3, 0)
    error += abs(i - check_i)
    #print("\t", i, " -> ", index_tuple, " -> ", check_i)
t2 = time.process_time()
cpu_time = t2 - t1
if (verbose): print("\nTotal error is: ", error)
if (verbose): print("Time elapsed is: ", cpu_time)
c_error += error
tot_time += cpu_time
tot_comp += test3Dmesh.tot_corners[0]
print("===============================================================================")
print("\n\nThe cumulative error is: ", c_error)

print("Total time elapsed for 3D grid: ", tot_time, "s")
print("Total number of function calls: ", tot_comp)
print("Time elapsed per function call: ", tot_time / tot_comp, "s")