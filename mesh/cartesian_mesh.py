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

reverse_order = False

# --------------------------------------------------------------------------- #
# Class definition
class cartesian_mesh_t:
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
        # Domain sizes
        self.domain_size = [None] * self.num_dims
        # Dimension orderings
        self.dimension_orderings = [range(0, self.num_dims), \
                                    range(self.num_dims - 1, -1, -1)]

        # Assigns number of cells and domain limits
        for i in range(0, self.num_dims):
            self.num_cells[i] = num_cells[i]
            self.domain[i][0] = domain[2*i]
            self.domain[i][1] = domain[2*i+1]
            self.domain_size[i] = self.domain[i][1] - self.domain[i][0]

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

        # Domain volume
        self.domain_volume = np.prod(self.domain_size)
        # Cell volume
        self.cell_volume   = np.prod(self.cell_size)

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
        (self.num_faces,   self.tot_faces,   self.num_face_orientations)   = self.num_points(1)
        # Edges
        (self.num_edges,   self.tot_edges,   self.num_edge_orientations)   = self.num_points(2)
        # Corners
        (self.num_corners, self.tot_corners, self.num_corner_orientations) = self.num_points(3)

        # Lists of lists (of lists)
        self.num_points = [[self.num_cells], self.num_faces, self.num_edges, self.num_corners]
        self.tot_points = [[self.tot_cells], self.tot_faces, self.tot_edges, self.tot_corners]
        self.tot_point_orientations = [1, self.num_face_orientations, self.num_edge_orientations, self.num_corner_orientations]

        # Computes cell face areas and edge lengths
        # Faces
        self.cell_faces_area = [1] * self.num_face_orientations
        for i in range(self.num_face_orientations):
            comb_idx = combination_index(self.num_dims, 1, i)
            for j in range(self.num_dims):
                if not j in comb_idx:
                    self.cell_faces_area[i] *= self.cell_size[j]
        # Edges
        self.cell_edges_length = [1] * self.num_edge_orientations
        for i in range(self.num_edge_orientations):
            comb_idx = combination_index(self.num_dims, 2, i)
            for j in range(self.num_dims):
                if not j in comb_idx:
                    self.cell_edges_length[i] *= self.cell_size[j]
        del(comb_idx)

        # Initializes coordinate fields
        # Cells
        self.cell_centre_array = self.cmp_coords()
        # Faces
        self.cell_face_arrays = [None] * self.num_face_orientations
        for i in range(self.num_face_orientations):
            self.cell_face_arrays[i] = self.cmp_coords(1, i)
        # Edges
        self.cell_edge_arrays = [None] * self.num_edge_orientations
        if self.num_dims > 1:
            for i in range(self.num_edge_orientations):
                self.cell_edge_arrays[i] = self.cmp_coords(2, i)
        # Corners
        self.cell_corner_arrays = [None] * self.num_corner_orientations
        if self.num_dims > 2:
            for i in range(self.num_corner_orientations):
                self.cell_corner_arrays[i] = self.cmp_coords(3, i)

        # Lists of lists (of lists)
        self.cell_coord_arrays = [[self.cell_centre_array], self.cell_face_arrays, \
                                   self.cell_edge_arrays, self.cell_corner_arrays]
        self.cell_dimensions   = [[self.cell_volume], self.cell_faces_area, self.cell_edges_length]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Spans dimensions in a circular way
    def rotate_dim(self, i, j = 1):
        k = i + j
        if (k >= self.num_dims):
            k %= self.num_dims
        return k

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Computes a total number of points (cells, faces, edges or corners)
    #                n = num_directions (n = 0, n = 1, n = 2 or n = 3  )
    def num_points(self, num_directions):
        if (self.num_dims > (num_directions - 1)):
            num_orientations = math.comb(self.num_dims, num_directions)
            num_points       = [None] * num_orientations
            tot_points       = [1] * num_orientations
            for i in range(0, num_orientations):
                num_points[i] = [None] * self.num_dims
                comb_idx = combination_index(self.num_dims, num_directions, i)
                for j in range(0, self.num_dims):
                    plus_one = 0
                    if j in comb_idx:
                        plus_one = 1
                    num_points[i][j] = (self.num_cells[j] + plus_one)
                    tot_points[i] *= num_points[i][j]
        else:
            num_orientations = 0
            num_points       = None
            tot_points       = None
        return (num_points, tot_points, num_orientations)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Computes a global index for (cells, faces, edges or corners)
    #          n = num_directions (n = 0, n = 1, n = 2 or n = 3  )
    def global_index(self, indices, num_directions = 0, orientation = 0):
        stride = 1
        if type(indices[0]) == int:
            index = 0
        elif type(indices[0]) == np.ndarray:
            index = np.zeros(np.size(indices[0]), dtype=int)
        for i in self.dimension_orderings[not reverse_order]:
            index = index + indices[i] * stride
            stride *= self.num_points[num_directions][orientation][i]
        return index

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Computes local index tuple for (cells, faces, edges or corners)
    #             n = num_directions (n = 0, n = 1, n = 2 or n = 3  )
    def local_index(self, index, num_directions = 0, orientation = 0):
        stride = self.tot_points[num_directions]
        if type(stride)==list: stride = stride[orientation]
        indices = [0] * self.num_dims
        for i in self.dimension_orderings[reverse_order]:
            stride = stride // self.num_points[num_directions][orientation][i]
            indices[i] = index // stride
            index = index % stride
        return tuple(indices)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Computes coordinates for (cells, faces, edges or corners)
    #       n = num_directions (n = 0, n = 1, n = 2 or n = 3  )
    def cmp_coords(self, num_directions = 0, orientation = 0):
        indexes  =  self.local_index(np.linspace(0, \
                    self.tot_points[num_directions][orientation]-1, \
                    self.tot_points[num_directions][orientation], dtype=int), \
                    num_directions, orientation)
        comb_idx = combination_index(self.num_dims, num_directions, orientation)
        if comb_idx==None: comb_idx=()
        coords = [0] * self.num_dims
        for i in range(self.num_dims):
            if i in comb_idx:
                x0 = self.domain[i][0]
            else:
                x0 = self.domain[i][0] + 0.5 * self.cell_size[i]
            coords[i] = x0 + self.cell_size[i] * indexes[i]
        return coords

