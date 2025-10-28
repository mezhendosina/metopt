
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
    
    # Count slack/surplus variables needed
    slack_var_count = sum(1 for _, ctype, _ in problem.constraints if ctype != ConstraintType.EQ)
    
    num_original_vars = problem.num_variables
    total_vars = num_original_vars + slack_var_count
    
    # Initialize constraint matrix and RHS vector
    constraint_matrix = np.zeros((len(problem.constraints), total_vars))
    rhs_vector = np.zeros(len(problem.constraints))
    
    current_slack_index = 0
    
    for i, (coeffs, ctype, rhs) in enumerate(problem.constraints):
        
        # Handle negative RHS by flipping the constraint
        if rhs < 0:
            constraint_coeffs = -coeffs
            rhs_val = -rhs
            if ctype == ConstraintType.LE:
                ctype = ConstraintType.GE
            elif ctype == ConstraintType.GE:
                ctype = ConstraintType.LE
            print(f"Constraint {i+1}: Multiplied by -1 (negative RHS)")
        else:
            constraint_coeffs = coeffs.copy()
            rhs_val = rhs
            
        # Copy original coefficients
        constraint_matrix[i, :num_original_vars] = constraint_coeffs
        rhs_vector[i] = rhs_val
        
        # Add slack/surplus variables
        if ctype == ConstraintType.LE:
            constraint_matrix[i, num_original_vars + current_slack_index] = 1.0
            print(f"Constraint {i+1}: Added slack variable s{current_slack_index + 1}")
            current_slack_index += 1
        elif ctype == ConstraintType.GE:
            constraint_matrix[i, num_original_vars + current_slack_index] = -1.0
            print(f"Constraint {i+1}: Added surplus variable s{current_slack_index + 1} (subtracted)")
            current_slack_index += 1
    
    print(f"\nCanonical form created with {num_original_vars} original variables")
    print(f"Total slack/surplus variables: {slack_var_count}")
    
    return objective_coeffs, constraint_matrix, rhs_vector
