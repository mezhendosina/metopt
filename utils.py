from typing import List, Optional
import numpy as np


def find_pivot_column(tableau: np.ndarray) -> Optional[int]:
    """
    Find pivot column using the most negative coefficient in objective row.
    Returns None if optimal solution is reached.
    """
    obj_row = tableau[0, :-1]  # Exclude RHS
    min_idx = np.argmin(obj_row)
    
    if obj_row[min_idx] < -1e-10:
        return min_idx
    return None


def find_pivot_row(tableau: np.ndarray, pivot_col: int) -> Optional[int]:
    """
    Find pivot row using minimum ratio test.
    Returns None if problem is unbounded.
    """
    ratios = []
    for i in range(1, len(tableau)):
        if tableau[i, pivot_col] > 1e-10:  # Positive coefficient
            ratio = tableau[i, -1] / tableau[i, pivot_col]
            if ratio >= 0:
                ratios.append((ratio, i))
    
    if not ratios:
        return None
    
    # Return row with minimum ratio
    return min(ratios, key=lambda x: x[0])[1]


def pivot_operation(tableau: np.ndarray, pivot_row: int, pivot_col: int, 
                   basis: List[int]) -> None:
    """
    Perform pivot operation on the tableau using NumPy operations.
    """
    pivot_element = tableau[pivot_row, pivot_col]
    
    # Divide pivot row by pivot element
    tableau[pivot_row] /= pivot_element
    
    # Eliminate pivot column in other rows
    for i in range(len(tableau)):
        if i != pivot_row:
            multiplier = tableau[i, pivot_col]
            tableau[i] -= multiplier * tableau[pivot_row]
    
    # Update basis
    basis[pivot_row - 1] = pivot_col
