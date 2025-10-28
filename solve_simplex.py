
from lp_problem import *
from main import find_pivot_row, pivot_operation
from utils import *


def solve_simplex(simplex_tableau: SimplexTableau, max_iterations: int = 100) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
    """
    Solve LP problem using Simplex method.
    
    Returns: (success, solution_vector, objective_value)
    """
    phase_name = "Phase I (Auxiliary Problem)" if simplex_tableau.is_auxiliary else "Phase II (Main Problem)"
    print(f"\n{'='*60}")
    print(f"STEP {'3' if simplex_tableau.is_auxiliary else '5'}: Solving {phase_name}")
    print(f"{'='*60}")
    
    iteration = 0
    simplex_tableau.print_tableau(iteration)
    
    while iteration < max_iterations:
        # Find pivot column
        pivot_col = find_pivot_column(simplex_tableau.tableau)
        
        if pivot_col is None:
            # Optimal solution found
            print(f"\nOptimal solution found at iteration {iteration}")
            
            # Extract solution
            solution = np.zeros(simplex_tableau.num_original_vars)
            for i, basis_var in enumerate(simplex_tableau.basis):
                if basis_var < simplex_tableau.num_original_vars:
                    solution[basis_var] = simplex_tableau.tableau[i + 1, -1]
            
            objective_value = simplex_tableau.tableau[0, -1]
            
            return True, solution, objective_value
        
        # Find pivot row
        pivot_row = find_pivot_row(simplex_tableau.tableau, pivot_col)
        
        if pivot_row is None:
            print("\nProblem is unbounded (no feasible solution with finite objective value)")
            return False, None, None
        
        print(f"\nIteration {iteration + 1}:")
        print(f"  Pivot column: {pivot_col}")
        print(f"  Pivot row: {pivot_row}")
        print(f"  Entering variable: x{pivot_col}")
        print(f"  Leaving variable: x{simplex_tableau.basis[pivot_row - 1]}")
        
        # Perform pivot operation
        pivot_operation(simplex_tableau.tableau, pivot_row, pivot_col, simplex_tableau.basis)
        
        iteration += 1
        simplex_tableau.print_tableau(iteration)
    
    print(f"\nMaximum iterations ({max_iterations}) reached")
    return False, None, None



def transition_to_main_problem(aux_tableau: SimplexTableau, 
                               original_objective: np.ndarray) -> Optional[SimplexTableau]:
    """
    Transition from auxiliary problem to main problem (Phase I to Phase II).
    """
    print("\n" + "="*60)
    print("STEP 4: Transitioning to Main Problem")
    print("="*60)
    
    # Check if auxiliary problem found feasible solution
    aux_obj_value = aux_tableau.tableau[0, -1]
    print(f"Auxiliary problem objective value: {aux_obj_value}")
    
    # Also check if any artificial variables are still in basis with non-zero values
    num_original_vars = aux_tableau.num_original_vars
    num_slack = len(aux_tableau.basis)
    artificial_start = num_original_vars + num_slack
    
    artificial_in_basis = False
    for i, basis_var in enumerate(aux_tableau.basis):
        if basis_var >= artificial_start:
            # Artificial variable in basis
            value = aux_tableau.tableau[i + 1, -1]
            if abs(value) > 1e-6:
                print(f"Artificial variable x{basis_var} is in basis with value {value}")
                artificial_in_basis = True
    
    if abs(aux_obj_value) > 1e-6 or artificial_in_basis:
        print("No feasible solution exists (auxiliary problem objective != 0 or artificial variables in basis)")
        print("This means the constraints are contradictory - feasible region is empty")
        return None
    
    print("Feasible solution found in auxiliary problem")
    
    # Remove artificial variables from tableau
    # Create new tableau without artificial variable columns
    new_num_cols = artificial_start + 1  # original + slack + RHS
    new_tableau = np.zeros((aux_tableau.num_rows, new_num_cols))
    
    # Create new objective row with original objective coefficients
    new_tableau[0, :num_original_vars] = -original_objective
    
    # Adjust objective row based on current basis
    for i, basis_var in enumerate(aux_tableau.basis):
        if basis_var < num_original_vars:
            # Basic variable in original variables
            coeff = original_objective[basis_var]
            new_tableau[0] += coeff * aux_tableau.tableau[i + 1, :new_num_cols]
    
    # Copy constraint rows (without artificial variable columns)
    new_tableau[1:, :artificial_start] = aux_tableau.tableau[1:, :artificial_start]
    new_tableau[1:, -1] = aux_tableau.tableau[1:, -1]
    
    print(f"Removed artificial variables")
    print(f"New tableau dimensions: {new_tableau.shape}")
    
    return SimplexTableau(new_tableau, aux_tableau.basis.copy(), 
                         num_original_vars, is_auxiliary=False)
