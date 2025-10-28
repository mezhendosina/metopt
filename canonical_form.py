
from typing import Tuple

import numpy as np
from lp_problem import *


def convert_to_canonical_form(problem: LPProblem) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert LP problem to canonical form:
    - All constraints are equalities
    - All variables are non-negative
    - RHS values are non-negative
    
    Returns: (objective_coeffs, constraint_matrix, rhs_vector)
    """
    print("\n" + "="*60)
    print("STEP 1: Converting to Canonical Form")
    print("="*60)
    
    # Convert minimization to maximization
    if problem.optimization_type == OptimizationType.MIN:
        objective_coeffs = -problem.objective_coefficients
        print("Converting MIN to MAX by negating objective coefficients")
    else:
        objective_coeffs = problem.objective_coefficients.copy()
    
    constraint_matrix = []
    rhs_vector = []
    slack_var_count = 0
    
    for i, (coeffs, ctype, rhs) in enumerate(problem.constraints):
        row = coeffs.copy()
        
        # Handle negative RHS
        if rhs < 0:
            row = -row
            rhs = -rhs
            # Flip constraint type
            if ctype == ConstraintType.LE:
                ctype = ConstraintType.GE
            elif ctype == ConstraintType.GE:
                ctype = ConstraintType.LE
            print(f"Constraint {i+1}: Multiplied by -1 (negative RHS)")
        
        # Add slack/surplus variables
        if ctype == ConstraintType.LE:
            slack_var_count += 1
            print(f"Constraint {i+1}: Added slack variable s{slack_var_count}")
        elif ctype == ConstraintType.GE:
            slack_var_count += 1
            print(f"Constraint {i+1}: Added surplus variable (subtracted)")
        
        constraint_matrix.append(row)
        rhs_vector.append(rhs)
    
    constraint_matrix = np.array(constraint_matrix)
    rhs_vector = np.array(rhs_vector)
    
    print(f"\nCanonical form created with {problem.num_variables} original variables")
    print(f"Total slack/surplus variables: {slack_var_count}")
    
    return objective_coeffs, constraint_matrix, rhs_vector
