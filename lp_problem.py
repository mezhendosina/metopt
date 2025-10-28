import numpy as np
from typing import List, Tuple, Optional


class OptimizationType(Enum):
    """Type of optimization: minimize or maximize"""
    MIN = "min"
    MAX = "max"


class ConstraintType(Enum):
    """Types of constraints"""
    LE = "<="  # Less than or equal
    GE = ">="  # Greater than or equal
    EQ = "="   # Equal


class LPProblem:
    """Represents a Linear Programming Problem"""
    
    def __init__(self):
        self.optimization_type: OptimizationType = OptimizationType.MAX
        self.objective_coefficients: np.ndarray = np.array([])
        self.constraints: List[Tuple[np.ndarray, ConstraintType, float]] = []
        self.variable_names: List[str] = []
        self.num_variables: int = 0
    
    def __str__(self):
        result = f"Optimization: {self.optimization_type.value}\n"
        result += f"Objective: {self.objective_coefficients}\n"
        result += f"Variables: {self.num_variables}\n"
        result += f"Constraints: {len(self.constraints)}\n"
        for i, (coeffs, ctype, rhs) in enumerate(self.constraints):
            result += f"  {i+1}: {coeffs} {ctype.value} {rhs}\n"
        return result


class SimplexTableau:
    """Represents a Simplex tableau for solving LP problems"""
    
    def __init__(self, tableau: np.ndarray, basis: List[int], 
                 num_original_vars: int, is_auxiliary: bool = False):
        self.tableau = tableau
        self.basis = basis
        self.num_original_vars = num_original_vars
        self.is_auxiliary = is_auxiliary
        self.num_rows, self.num_cols = tableau.shape
    
    def print_tableau(self, iteration: int = 0):
        """Print the current tableau state"""
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}")
        print(f"{'='*60}")
        print(f"Basis: {self.basis}")
        print("\nTableau:")
        for i in range(self.num_rows):
            if i == 0:
                print("Z  |", end="")
            else:
                print(f"x{self.basis[i-1]:2d}|", end="")
            for val in self.tableau[i]:
                print(f"{val:8.3f}", end=" ")
            print()
        print(f"{'='*60}\n")
