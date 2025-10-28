
import sys
from lp_problem import *


def read_lp_problem(filename: str) -> LPProblem:
    """
    Read LP problem from file.
    
    File format:
    Line 1: optimization_type (min or max)
    Line 2: number_of_variables
    Line 3: objective_coefficients (space-separated)
    Line 4: number_of_constraints
    Following lines: constraint_coefficients constraint_type rhs
    
    Example:
    max
    2
    3 5
    3
    1 0 <= 4
    0 2 <= 12
    3 2 <= 18
    """
    problem = LPProblem()
    
    try:
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            
            # Read optimization type
            opt_type = lines[0].lower()
            problem.optimization_type = OptimizationType.MAX if opt_type == 'max' else OptimizationType.MIN
            
            # Read number of variables
            problem.num_variables = int(lines[1])
            problem.variable_names = [f"x{i+1}" for i in range(problem.num_variables)]
            
            # Read objective coefficients
            problem.objective_coefficients = np.array(list(map(float, lines[2].split())))
            
            # Read number of constraints
            num_constraints = int(lines[3])
            
            # Read constraints
            for i in range(4, 4 + num_constraints):
                parts = lines[i].split()
                coeffs = np.array(list(map(float, parts[:problem.num_variables])))
                constraint_type_str = parts[problem.num_variables]
                rhs = float(parts[problem.num_variables + 1])
                
                if constraint_type_str == "<=":
                    ctype = ConstraintType.LE
                elif constraint_type_str == ">=":
                    ctype = ConstraintType.GE
                else:
                    ctype = ConstraintType.EQ
                
                problem.constraints.append((coeffs, ctype, rhs))
        
        return problem
    
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)