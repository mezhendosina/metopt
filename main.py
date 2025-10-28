import sys

from auxilaty_problem import create_auxiliary_problem
from canonical_form import convert_to_canonical_form
from lp_problem import *
from read_file import read_lp_problem
from solve_simplex import solve_simplex, transition_to_main_problem



def solve_lp_problem(filename: str, output_filename: str = None) -> None:
    """
    Main function to solve LP problem - complete cycle.
    """
    print("\n" + "="*80)
    print("LINEAR PROGRAMMING PROBLEM SOLVER")
    print("="*80)
    
    # Step 1: Read problem from file
    print(f"\nReading problem from: {filename}")
    problem = read_lp_problem(filename)
    print("\nOriginal Problem:")
    print(problem)
    
    # Step 2: Convert to canonical form
    objective_coeffs, constraint_matrix, rhs_vector = convert_to_canonical_form(problem)
    
    # Step 3: Create and solve auxiliary problem
    aux_tableau = create_auxiliary_problem(objective_coeffs, constraint_matrix, 
                                          rhs_vector, problem)
    
    success, aux_solution, aux_objective = solve_simplex(aux_tableau)
    
    if not success:
        result = "No solution: Auxiliary problem failed (feasible region may be empty)"
        print(f"\n{'='*80}")
        print("FINAL RESULT")
        print(f"{'='*80}")
        print(result)
        if output_filename:
            with open(output_filename, 'w') as f:
                f.write(result)
        return
    
    # Step 4: Transition to main problem
    main_tableau = transition_to_main_problem(aux_tableau, objective_coeffs)
    
    if main_tableau is None:
        result = "No solution: Feasible region is empty"
        print(f"\n{'='*80}")
        print("FINAL RESULT")
        print(f"{'='*80}")
        print(result)
        if output_filename:
            with open(output_filename, 'w') as f:
                f.write(result)
        return
    
    # Step 5: Solve main problem
    success, solution, objective_value = solve_simplex(main_tableau)
    
    # Step 6: Output result
    print(f"\n{'='*80}")
    print("FINAL RESULT")
    print(f"{'='*80}")
    
    if not success:
        result = "No solution: Problem is unbounded or infeasible"
        print(result)
    else:
        # The objective value from the tableau is correct for minimization problems
        # that have been converted to maximization. No final sign flip is needed.
        
        result = f"Optimal solution found!\n\n"
        result += f"Optimization type: {problem.optimization_type.value}\n"
        result += f"Optimal point: x* = ("
        result += ", ".join([f"{val:.6f}" for val in solution])
        result += ")\n"
        result += f"Objective function value: f(x*) = {objective_value:.6f}\n"
        
        print(result)
        
        # Detailed output
        print("\nDetailed solution:")
        for i, val in enumerate(solution):
            print(f"  x{i+1} = {val:.6f}")
    
    # Write to output file if specified
    if output_filename:
        with open(output_filename, 'w') as f:
            f.write(result)
        print(f"\nResult written to: {output_filename}")
    
    print(f"{'='*80}\n")

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    solve_lp_problem(input_file, output_file)