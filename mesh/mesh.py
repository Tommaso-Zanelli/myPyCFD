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
# Modules
import numpy as np
import math
import sys
sys.path.append('../')
from tools.combination_index import combination_index

# --------------------------------------------------------------------------- #
# Class definition
class mesh:
    """A class containing details of a Cartesian mesh and indexing functions."""

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # 
    # Constructor
    def __init__(self, domain, num_cells, is_periodic = None):

        # Number of dimensions of the domain
        self.num_dims  = len(num_cells)
        # Number of cells along each dimension (list)
        self.num_cells = [None] * self.num_dims
        # Domain limits along each dimension (list of lists)
        self.domain    = [[None] * 2] * self.num_dims

        # Assigns number of cells and domain limits 
        for i in range(0, self.num_dims):
            self.num_cells[i] = num_cells[i]
            self.domain[i][0] = domain[2*i]
            self.domain[i][1] = domain[2*i+1]

        # Cell centre coordinates (list of numpy arrays)
        self.cell_centres = [None] * self.num_dims
        # Cell face coordinates (list of numpy arrays)
        self.cell_faces   = [None] * self.num_dims
        # Cell spacings
        self.cell_size = [None] * self.num_dims

        # Assigns cell centres
        for i in range(0, self.num_dims):
            self.cell_size[i]    = (self.domain[i][1] - self.domain[i][0]) / self.num_cells[i]
            self.cell_centres[i] = np.linspace(self.domain[i][0] + 0.5 * self.cell_size[i], \
                                               self.domain[i][1] - 0.5 * self.cell_size[i], \
                                               self.num_cells[i])
            self.cell_faces[i]   = np.linspace(self.domain[i][0], self.domain[i][1], \
                                               (self.num_cells[i] + 1))

        # Assigns the periodicity of the domain boundaries
        self.is_periodic = [[False] * 2] * self.num_dims
        if is_periodic != None:
            for i in range(0, self.num_dims):
                self.is_periodic[i][0] = is_periodic[2*i]
                self.is_periodic[i][1] = is_periodic[2*i+1]

        # Assigns numbers of cells, faces, edges, corners.
        # Cells
        self.tot_cells = math.prod(self.num_cells)
        # Faces
        (self.num_faces,   self.num_face_orientations)   = self.num_points(1)
        # Edges
        (self.num_edges,   self.num_edge_orientations)   = self.num_points(2)
        # Corners
        (self.num_corners, self.num_corner_orientations) = self.num_points(3)


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # 
    # Spans dimensions in a circular way          
    def rotate_dim(self, i, j = 1):
        k = i + j
        if (k >= self.num_dims):
            k %= self.num_dims
        return k

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # 
    # Computes a total number of points (cells, faces, edges or corners)
    #                                   (n = 0, n = 1, n = 2 or n = 3  )
    def num_points(self, n):
        if (self.num_dims > (n - 1)):
            num_orientations = math.comb(self.num_dims, n)
            num_points       = [1] * num_orientations
            for i in range(0, num_orientations):
                rotations = [None] * n
                for j in range(0, n):
                    rotations[j] = self.rotate_dim(i, j)
                for j in range(0, self.num_dims):
                    plus_one = 0
                    if j in rotations:
                        plus_one = 1
                    num_points[i] *= (self.num_cells[j] + plus_one)
        else:
            num_orientations = None
            num_points       = None
        return (num_points, num_orientations)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # 
    # Computes a global index for (cells, faces, edges or corners)
    #                             (n = 0, n = 1, n = 2 or n = 3  )
    def global_index(self, indices, n = 0, orientation = 0):
        index  = 0
        stride = 1
        rotations = [None] * n
        for j in range(0, n):
            rotations[j] = self.rotate_dim(orientation, j)
        for i in range(0, self.num_dims):
            index += indices[i] * stride
            plus_one = 0
            if i in rotations:
                plus_one = 1
            stride *= (self.num_cells[i] + plus_one)

