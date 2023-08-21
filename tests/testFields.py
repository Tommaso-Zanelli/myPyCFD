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
import time # check if actually needed
import sys
import numpy as np # check if actually needed
sys.path.append('../')
from mesh.cartesian_mesh import cartesian_mesh_t
from tools.combination_index import combination_index # check if actually needed
from fields.field import field_t

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
# Test 2D mesh field initialization
test2Dmesh = cartesian_mesh_t((0, 1, 0, 1), (Nx_2D, Ny_2D))
field2D_1 = field_t(test2Dmesh)
field2D_2 = field_t(test2Dmesh, 9)
def f0(xx):
    return xx[0] ** 2
field2D_3 = field_t(test2Dmesh, f0, 1, 1)
field2D_4 = field2D_1.create_copy()