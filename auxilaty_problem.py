import numpy as np

from lp_problem import *


def create_auxiliary_problem(objective_coeffs: np.ndarray, 
                            constraint_matrix: np.ndarray, 
                            rhs_vector: np.ndarray,
                            original_problem: LPProblem) -> SimplexTableau:
    """
    Create auxiliary problem (Phase I) to find initial feasible solution.
    Adds artificial variables and creates initial tableau.
    """
    print("\n" + "="*60)
    print("STEP 2: Creating Auxiliary Problem (Phase I)")
    print("="*60)
    
    num_constraints = len(constraint_matrix)
    num_original_vars = len(objective_coeffs)
    
    # Determine which constraints need artificial variables
    artificial_needed = []
    slack_positions = []
    current_slack = num_original_vars
    
    for i, (coeffs, ctype, rhs) in enumerate(original_problem.constraints):
        if ctype == ConstraintType.LE:
            slack_positions.append(current_slack)
            current_slack += 1
            artificial_needed.append(False)
        elif ctype == ConstraintType.GE:
            slack_positions.append(current_slack)
            current_slack += 1
            artificial_needed.append(True)
        else:  # EQ
            slack_positions.append(-1)
            artificial_needed.append(True)
    
    num_artificial = sum(artificial_needed)
    print(f"Artificial variables needed: {num_artificial}")
    
    # Build initial tableau for auxiliary problem
    num_slack = len([x for x in slack_positions if x >= 0])
    total_vars = num_original_vars + num_slack + num_artificial
    
    # Initialize tableau: [objective_row; constraint_rows]
    tableau = np.zeros((num_constraints + 1, total_vars + 1))
    
    # Constraint rows
    basis = []
    artificial_col = num_original_vars + num_slack
    artificial_idx = 0
    
    for i in range(num_constraints):
        # Original variable coefficients
        tableau[i + 1, :num_original_vars] = constraint_matrix[i]
        
        # Slack/surplus variable
        if slack_positions[i] >= 0:
            if original_problem.constraints[i][1] == ConstraintType.LE:
                tableau[i + 1, slack_positions[i]] = 1.0
                basis.append(slack_positions[i])
            else:  # GE
                tableau[i + 1, slack_positions[i]] = -1.0
        
        # Artificial variable if needed
        if artificial_needed[i]:
            tableau[i + 1, artificial_col + artificial_idx] = 1.0
            basis.append(artificial_col + artificial_idx)
            # Update objective row (minimize sum of artificial variables)
            tableau[0, :total_vars] -= tableau[i + 1, :total_vars]
            tableau[0, -1] -= rhs_vector[i]
            artificial_idx += 1
        
        # RHS
        tableau[i + 1, -1] = rhs_vector[i]
    
    print(f"Initial basis: {basis}")
    print(f"Tableau dimensions: {tableau.shape}")
    
    return SimplexTableau(tableau, basis, num_original_vars, is_auxiliary=True)


