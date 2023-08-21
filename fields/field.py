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
import sys
sys.path.append('../')
from mesh.cartesian_mesh import cartesian_mesh_t
from tools.combination_index import combination_index # check if actually needed

# --------------------------------------------------------------------------- #
# Class definition
class field_t:
    """A class containing a scalar field over a Cartesian mesh."""

    # ----------------------------------------------------------------------- #
    # Constructor method

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Constructor for a field over mesh elements (cells, faces, edges or corners)
    #                         n = num_directions (n = 0, n = 1, n = 2 or n = 3  )
    def __init__(self, mesh, init_values=0.0, num_directions=0, orientation=0):

        # Assigns the mesh to the field
        self.mesh = mesh

        # Assigns the position of the field in the mesh
        self.num_directions = num_directions
        self.orientation    = orientation

        # Assigns total number of values in the field
        self.tot_points = self.mesh.tot_points[num_directions][orientation]

        # Initializes the field
        if callable(init_values):
            # If the initial condition is a function or method, pass to it the mesh coordinates
            coords_temp = self.mesh.cell_coord_arrays[num_directions][orientation]
            self.values = init_values(tuple(coords_temp))

        elif type(init_values) == int or type(init_values) == float:
            # If the initial condition is a constant value, assign it to the array
            self.values = np.ones(self.tot_points, dtype=np.float64) * init_values

        elif type(init_values)==np.ndarray:
            # If the initial condiiton is already a numpy array, check its shape and assign it
            if len(np.shape(init_values)) == 1 and np.size(init_values, 0) == self.tot_points:
                self.values = init_values
            else:
                self.values = None
                print("ERROR: inconsistent field shape (", np.shape(init_values), \
                      " vs. (", self.mesh.tot_points[num_directions][orientation], ",))")

        else:
            # Default is an error message
            self.values = None
            print("ERROR: invald initial condition: type = ", type(init_values))

    # ----------------------------------------------------------------------- #
    # Overload indexing ("[]") operator
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Overload get item operator
    def __getitem__(self, key):
        if isinstance(key, slice):
            print("Slicing not implemented yet")
            return None
        if type(key)==int:
            return self.values[key]
        elif type(key) == tuple:
            return self.values[self.mesh.global_index(key, self.num_directions, self.orientation)]
        else:
            print("Unrecognised index type: ", type(key))
            return None

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Overload set item operator
    def __setitem__(self, key, value):
        if isinstance(key, slice):
            print("Slicing not implemented yet")
            return None
        if type(key)==int:
            self.values[key] = value
        elif type(key) == tuple:
            self.values[self.mesh.global_index(key, self.num_directions, self.orientation)] = value
        else:
            print("Unrecognised index type: ", type(key))


    # ----------------------------------------------------------------------- #
    # Overload binary operators

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Create copy of the current object and return it
    def create_copy(self):
        copy_obj = field_t(self.mesh, self.values, self.num_directions, self.orientation)
        return copy_obj

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Overload addition operator
    def __add__(self, other):
        copy_obj = self.create_copy()
        if type(other) == int:
            copy_obj.values = self.values + float(other)
        elif type(other) == float:
            copy_obj.values = self.values + other
        elif type(other) == np.ndarray:
            if len(np.shape(other)) == 1 and np.size(other, 0) == self.tot_points:
                copy_obj.values = self.values + other
            else:
                print("ERROR: inconsistent field shape (", np.shape(other), " vs. (", \
                      self.mesh.tot_points[self.num_directions][self.orientation], ",))")
                copy_obj = None                   
        elif type(other) == field_t:
            if other.tot_points == self.tot_points:
                copy_obj.values = self.values + other.values
            else:
                print("ERROR: inconsistent field size (", other.tot_points," vs. ", self.tot_points, ")")
                copy_obj = None
        else:
            print("ERROR: unknown operand type: ", type(other))
            copy_obj = None
        return copy_obj

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Overload subtraction operator
    def __sub__(self, other):
        copy_obj = self.create_copy()
        if type(other) == int:
            copy_obj.values = self.values - float(other)
        elif type(other) == float:
            copy_obj.values = self.values - other
        elif type(other) == np.ndarray:
            if len(np.shape(other)) == 1 and np.size(other, 0) == self.tot_points:
                copy_obj.values = self.values - other
            else:
                print("ERROR: inconsistent field shape (", np.shape(other), " vs. (", \
                      self.mesh.tot_points[self.num_directions][self.orientation], ",))")
                copy_obj = None                   
        elif type(other) == field_t:
            if other.tot_points == self.tot_points:
                copy_obj.values = self.values - other.values
            else:
                print("ERROR: inconsistent field size (", other.tot_points," vs. ", self.tot_points, ")")
                copy_obj = None
        else:
            print("ERROR: unknown operand type: ", type(other))
            copy_obj = None
        return copy_obj

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Overload multiplication operator
    def __mul__(self, other):
        copy_obj = self.create_copy()
        if type(other) == int:
            copy_obj.values = self.values * float(other)
        elif type(other) == float:
            copy_obj.values = self.values * other
        elif type(other) == np.ndarray:
            if len(np.shape(other)) == 1 and np.size(other, 0) == self.tot_points:
                copy_obj.values = self.values * other
            else:
                print("ERROR: inconsistent field shape (", np.shape(other), " vs. (", \
                      self.mesh.tot_points[self.num_directions][self.orientation], ",))")
                copy_obj = None                   
        elif type(other) == field_t:
            if other.tot_points == self.tot_points:
                copy_obj.values = self.values * other.values
            else:
                print("ERROR: inconsistent field size (", other.tot_points," vs. ", self.tot_points, ")")
                copy_obj = None
        else:
            print("ERROR: unknown operand type: ", type(other))
            copy_obj = None
        return copy_obj

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Overload matmul ("@") operator to perform dot multiplication
    def __matmul__(self, other):
        result
        if type(other) == np.ndarray:
            if len(np.shape(other)) == 1 and np.size(other, 0) == self.tot_points:
                result = self.values @ other
            else:
                print("ERROR: inconsistent field shape (", np.shape(other), " vs. (", \
                      self.mesh.tot_points[self.num_directions][self.orientation], ",))")
                result = None                   
        elif type(other) == field_t:
            if other.tot_points == self.tot_points:
                result = self.values @ other.values
            else:
                print("ERROR: inconsistent field size (", other.tot_points," vs. ", self.tot_points, ")")
                result = None
        else:
            print("ERROR: unknown operand type: ", type(other))
            result = None
        return result

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Overload truediv ("/") operator
    def __truediv__(self, other):
        copy_obj = self.create_copy()
        if type(other) == int:
            copy_obj.values = self.values / float(other)
        elif type(other) == float:
            copy_obj.values = self.values / other
        elif type(other) == np.ndarray:
            if len(np.shape(other)) == 1 and np.size(other, 0) == self.tot_points:
                copy_obj.values = self.values / other
            else:
                print("ERROR: inconsistent field shape (", np.shape(other), " vs. (", \
                      self.mesh.tot_points[self.num_directions][self.orientation], ",))")
                copy_obj = None                   
        elif type(other) == field_t:
            if other.tot_points == self.tot_points:
                copy_obj.values = self.values / other.values
            else:
                print("ERROR: inconsistent field size (", other.tot_points," vs. ", self.tot_points, ")")
                copy_obj = None
        else:
            print("ERROR: unknown operand type: ", type(other))
            copy_obj = None
        return copy_obj

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Overload floordiv ("//") operator
    def __floordiv__(self, other):
        copy_obj = self.create_copy()
        if type(other) == int:
            copy_obj.values = self.values // float(other)
        elif type(other) == float:
            copy_obj.values = self.values // other
        elif type(other) == np.ndarray:
            if len(np.shape(other)) == 1 and np.size(other, 0) == self.tot_points:
                copy_obj.values = self.values // other
            else:
                print("ERROR: inconsistent field shape (", np.shape(other), " vs. (", \
                      self.mesh.tot_points[self.num_directions][self.orientation], ",))")
                copy_obj = None                   
        elif type(other) == field_t:
            if other.tot_points == self.tot_points:
                copy_obj.values = self.values // other.values
            else:
                print("ERROR: inconsistent field size (", other.tot_points," vs. ", self.tot_points, ")")
                copy_obj = None
        else:
            print("ERROR: unknown operand type: ", type(other))
            copy_obj = None
        return copy_obj

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Overload mod ("%") operator
    def __mod__(self, other):
        copy_obj = self.create_copy()
        if type(other) == int:
            copy_obj.values = self.values % float(other)
        elif type(other) == float:
            copy_obj.values = self.values % other
        elif type(other) == np.ndarray:
            if len(np.shape(other)) == 1 and np.size(other, 0) == self.tot_points:
                copy_obj.values = self.values % other
            else:
                print("ERROR: inconsistent field shape (", np.shape(other), " vs. (", \
                      self.mesh.tot_points[self.num_directions][self.orientation], ",))")
                copy_obj = None                   
        elif type(other) == field_t:
            if other.tot_points == self.tot_points:
                copy_obj.values = self.values % other.values
            else:
                print("ERROR: inconsistent field size (", other.tot_points," vs. ", self.tot_points, ")")
                copy_obj = None
        else:
            print("ERROR: unknown operand type: ", type(other))
            copy_obj = None
        return copy_obj

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Overload pow ("**") operator
    def __pow__(self, other):
        copy_obj = self.create_copy()
        if type(other) == int:
            copy_obj.values = self.values ** other
        elif type(other) == float:
            copy_obj.values = self.values ** other
        elif type(other) == np.ndarray:
            if len(np.shape(other)) == 1 and np.size(other, 0) == self.tot_points:
                copy_obj.values = self.values ** other
            else:
                print("ERROR: inconsistent field shape (", np.shape(other), " vs. (", \
                      self.mesh.tot_points[self.num_directions][self.orientation], ",))")
                copy_obj = None                   
        elif type(other) == field_t:
            if other.tot_points == self.tot_points:
                copy_obj.values = self.values ** other.values
            else:
                print("ERROR: inconsistent field size (", other.tot_points," vs. ", self.tot_points, ")")
                copy_obj = None
        else:
            print("ERROR: unknown operand type: ", type(other))
            copy_obj = None
        return copy_obj

    # ----------------------------------------------------------------------- #
    # Overload binary operators (inverted operand versions)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Overload addition operator
    def __radd__(self, other):
        return self.__add__(other)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Overload subtraction operator
    def __rsub__(self, other):
        copy_obj = self.create_copy()
        if type(other) == int:
            copy_obj.values = float(other) - self.values
        elif type(other) == float:
            copy_obj.values = other - self.values
        elif type(other) == np.ndarray:
            if len(np.shape(other)) == 1 and np.size(other, 0) == self.tot_points:
                copy_obj.values = other - self.values
            else:
                print("ERROR: inconsistent field shape (", np.shape(other), " vs. (", \
                      self.mesh.tot_points[self.num_directions][self.orientation], ",))")
                copy_obj = None                   
        elif type(other) == field_t:
            if other.tot_points == self.tot_points:
                copy_obj.values = other.values - self.values
            else:
                print("ERROR: inconsistent field size (", other.tot_points," vs. ", self.tot_points, ")")
                copy_obj = None
        else:
            print("ERROR: unknown operand type: ", type(other))
            copy_obj = None
        return copy_obj

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Overload multiplication operator
    def __rmul__(self, other):
        return self.__mul__(other)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Overload matmul ("@") operator to perform dot multiplication
    def __rmatmul__(self, other):
        return self.__matmul__(other)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Overload truediv ("/") operator
    def __rtruediv__(self, other):
        return None

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Overload floordiv ("//") operator
    def __rfloordiv__(self, other):
        return None

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Overload mod ("%") operator
    def __rmod__(self, other):
        return None

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Overload pow ("**") operator
    def __rpow__(self, other):
        copy_obj = self.create_copy()
        if type(other) == int:
            copy_obj.values = float(other) ** self.values
        elif type(other) == float:
            copy_obj.values = other ** self.values
        elif type(other) == np.ndarray:
            if len(np.shape(other)) == 1 and np.size(other, 0) == self.tot_points:
                copy_obj.values = other ** self.values
            else:
                print("ERROR: inconsistent field shape (", np.shape(other), " vs. (", \
                      self.mesh.tot_points[self.num_directions][self.orientation], ",))")
                copy_obj = None                   
        elif type(other) == field_t:
            if other.tot_points == self.tot_points:
                copy_obj.values = other.values ** self.values
            else:
                print("ERROR: inconsistent field size (", other.tot_points," vs. ", self.tot_points, ")")
                copy_obj = None
        else:
            print("ERROR: unknown operand type: ", type(other))
            copy_obj = None
        return copy_obj

    # ----------------------------------------------------------------------- #
    # Overload unary operators

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Overload addition operator
    def __iadd__(self, other):
        pass

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Overload subtraction operator
    def __isub__(self, other):
        pass

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Overload multiplication operator
    def __imul__(self, other):
        pass

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Overload truediv ("/") operator
    def __itruediv__(self, other):
        pass

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Overload floordiv ("//") operator
    def __ifloordiv__(self, other):
        pass

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Overload mod ("%") operator
    def __imod__(self, other):
        pass

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Overload pow ("**") operator
    def __ipow__(self, other):
        pass