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
from math import comb

def combination_index(num_dimensions, num_directions, combination = None):
    """
    Parameters
    ----------
    num_dimensions : integer
        The total number of dimensions.
    num_directions : integer
        The number of directions considered for combinations.
    combination : integer or tuple of integers, optional.
        If integer, returns the correspoinding combination of directions.
        If tuple, returns the corresponding index.
        If None, returns all combinations in order.
        The default is None.
    """
    if num_directions > num_dimensions:
            print("ERROR: number of directions cannot be greater than the number of dimensions!")
            return None
    if type(combination) is tuple:
        directions_list = list(combination)
        directions_list.sort()
        if len(directions_list) != num_directions:
            print("ERROR: inconsistent number of directions!")
            return None
    elif combination is None:
        global_index_list = []
        dir_indexes_list  = []
    num_combinations = comb(num_dimensions, num_directions)
    global_index = 0
    dir_indexes  = list(range((num_directions - 1), -1, -1))
    current_dir = 0
    while current_dir < num_directions:
        if type(combination) is tuple:
            if directions_list == dir_indexes[::-1]: return global_index
        elif type(combination) is int:
            if combination == global_index: return tuple(dir_indexes[::-1])
        elif combination is None:
                global_index_list.append(global_index)
                dir_indexes_list.append(tuple(dir_indexes[::-1]))
        validIncrease = False
        while (not validIncrease) and (current_dir < num_directions):
            dir_indexes[current_dir]+=1
            for subsequentDirs in range(current_dir-1, -1, -1):
                dir_indexes[subsequentDirs] = dir_indexes[subsequentDirs + 1] + 1
            if any(value >= num_dimensions for value in dir_indexes):
                current_dir += 1
            else:
                current_dir = 0
                validIncrease = True
        global_index += 1
    if combination is None:
        return (tuple(global_index_list), tuple(dir_indexes_list))
    return None
