"""Core module for solution represation in MutaPy

This module contains the base structure that allows nature-inspired 
algorithms to operate in continuous search spaces. It acts as a bridge 
between topological representations (graphs) and vector algebraic operations.
"""

from dataclasses import dataclass
import numpy as np
from typing import Callable, Optional


@dataclass(init=True, repr=True)
class Solution:
    """Generic class to model the solutions for the optimization problem.

    `Solution` abstracts the problem domain into a numerical vector
    (`attribute_list`) that nature inspired algorithms (`NIA`) can operate.
    It requires injecting external evaluation and mutation functions to
    decouple the algorithm's physics from the problem's logic 

    Attributes:
        attribute_list (np.array):  Multidimensional array with the current
            features of the solution (e.g., edge probabilities, chess squares, etc.).
        interval_by_attr (np.array): 2D array of shape (N, 2) with the minimum
            and maximo allowed values for each respective attribute.
        _fitness_function (Callable): Injected function to evaluate the solution.
            Must accept a `Solution` instance and return a `float`.
        _random_function (Callable): Injected function to randomize the solution.
            Should affect the `attribute_list` attribute.
        _show_function (Callable): Injected function to print the solution. 
            Should return a string to be printed.
        _mutation_function (Callable): Injected function to mutate the solution.
            Should affect the `attribute_list` attribute.
        _add_funct(Optional[Callable]): Injected operator for `Solution` addition. 
        _sub_funct(Optional[Callable]): Injected operator for `Solution` subtraction.
        _mul_funct(Optional[Callable]): Injected operator for `Solution` product.
        
    Example:
    >>> import numpy as np
    >>> initial_sol = np.array([0.2, 0.3, 0.2, 0.3, 0.3])
    >>> limits_sol = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    >>> sol = Solution(initial_dna, limits, my_fitness, my_mutation, ...)
    >>> sol.correct_solution_limits(float)
    """
    attribute_list: np.array
    interval_by_attr: np.array

    _fitness_function: Callable
    _random_function: Callable
    _show_function: Callable
    _mutation_function: Callable

    _add_funct: Callable = None
    _sub_funct: Callable = None
    _mul_funct: Callable = None
    _div_funct: Callable = None

    def __add__(self, other:'Solution') -> 'Solution':
        """Overload of the addition (+) operator.

        Delegates the algebraic operation to the function injected in `_add_funct`.

        Args:
            other (Solution): the other solution to add.

        Returns:
            Solution: A new instance with the resulting `Solution`.
            
        Raises:
            NotImplementedError: If `_add_funct`has not been injected.
        """
        
        if self._add_funct is None:
            raise NotImplementedError("Addition operator not injected in solution.")
        return self._add_funct(self, other) 

    def __sub__(self, other:'Solution') -> 'Solution':
        """Overload of the subtraction (-) operator.

        Delegates the algebraic operation to the function injected in `_sub_funct`.

        Args:
            other (Solution): the other solution to subtract.

        Returns:
            Solution: A new instance with the resulting `Solution`.
            
        Raises:
            NotImplementedError: If `_sub_funct`has not been injected.
        """
        if self._sub_funct is None:
            raise NotImplementedError("Subtraction operator not injected in solution.")
        return self._sub_funct(self, other) 

    def __mul__(self, other:'Solution') -> 'Solution':
        """Overload of the product (*) operator.

        Delegates the algebraic operation to the function injected in `_mul_funct`.

        Args:
            other (Solution): the other solution to multiply.

        Returns:
            Solution: A new instance with the resulting `Solution`.
            
        Raises:
            NotImplementedError: If `_mul_funct`has not been injected.
        """
        if self._sub_funct is None:
            raise NotImplementedError("Product operator not injected in solution.")
        return self._mul_funct(self, other) 

    def fitness_function(self) -> float:
        """Calculates the fitness value of the current solution.

        Returns:
            float: The error or cost calculated by `_fitness_function` function.
        """
        return self._fitness_function(self)

    def mutation_function(self) -> np.array:
        """Mutates the attributes from the current solution.

        Returns:
            np.array: The resulting new attributes after the mutation.
        """
        return self._mutation_function(self) 
    
    def randomize_function(self) -> np.array:
        """Randomize the attributes from the current solution.

        Returns:
            np.array: The resulting new attributes after the randomization.
        """
        return self._random_function(self)

    def show_solution(self) -> str:
        """Print in a custom way the attributes from the current solution.

        Returns:
            str: The string result from execute `_show_function`.
        """
        return self._show_function(self)

    def correct_solution_limits(self, type_data: type = float) -> None:
        """Corrects the vector values to ensure they are within bounds.

        Iterates through `attribute_list` and clips any value that exceeds 
        the boundaries defined in `interval_by_attr`. It also ensures the 
        array's data type is correct.

        Args:
            type_data (type, optional): The data type to cast the resulting 
                array to. Defaults to `float`.

        Raises:
            ValueError: If the dimensions of `attribute_list` and 
                `interval_by_attr` do not match.
        """
        if len(self.attribute_list) != len(self.interval_by_attr):
            raise ValueError("Dimensions of attributes and intervals do not match.")
        
        minimums = self.interval_by_attr[:, 0]
        maximums = self.interval_by_attr[:, 1]
        
        self.attribute_list = np.clip(self.attribute_list, minimums, maximums)
        self.attribute_list = self.attribute_list.astype(type_data)
