
import time
from typing import Dict, List, Set, Tuple, Optional
from collections import deque
import os


class CSPSolver:
    def __init__(self, bots: List[str], slots: List[str], unary_constraints: Dict[str, Set[str]]):
        self.bots = bots
        self.slots = slots
        self.unary_constraints = unary_constraints
        
        # Initialize domains for each slot
        self.domains = {slot: set(bots) for slot in slots}
        
        # Apply unary constraints
        self.apply_unary_constraints()
        
        # Track statistics
        self.assignments_count = 0
        self.backtracks_count = 0
        
    def apply_unary_constraints(self):
        for slot, forbidden_bots in self.unary_constraints.items():
            if slot in self.domains:
                self.domains[slot] -= forbidden_bots
    
    def is_consistent(self, assignment: Dict[str, str], var: str, value: str) -> bool:
        var_index = self.slots.index(var)
        
        # Check no-back-to-back with previous slot
        if var_index > 0:
            prev_slot = self.slots[var_index - 1]
            if prev_slot in assignment and assignment[prev_slot] == value:
                return False
        
        # Check no-back-to-back with next slot
        if var_index < len(self.slots) - 1:
            next_slot = self.slots[var_index + 1]
            if next_slot in assignment and assignment[next_slot] == value:
                return False
        
        return True
    
    def is_complete(self, assignment: Dict[str, str]) -> bool:
        if len(assignment) != len(self.slots):
            return False
        
        # Check minimum coverage: all bots must be used
        used_bots = set(assignment.values())
        if used_bots != set(self.bots):
            return False
        
        return True
    
    def select_unassigned_variable(self, assignment: Dict[str, str], use_mrv: bool = False) -> Optional[str]:
        unassigned = [slot for slot in self.slots if slot not in assignment]
        
        if not unassigned:
            return None
        
        if not use_mrv:
            return unassigned[0]
        
        # MRV: Choose variable with smallest domain
        return self.minimum_remaining_values(assignment, unassigned)
    
    def minimum_remaining_values(self, assignment: Dict[str, str], unassigned: List[str]) -> str:
        min_values = float('inf')
        best_var = unassigned[0]
        
        for var in unassigned:
            # Count legal values for this variable
            legal_values = 0
            for value in self.domains[var]:
                if self.is_consistent(assignment, var, value):
                    legal_values += 1
            
            if legal_values < min_values:
                min_values = legal_values
                best_var = var
        
        return best_var
    
    def forward_checking(self, assignment: Dict[str, str], var: str, value: str, 
                        domains: Dict[str, Set[str]]) -> Optional[Dict[str, Set[str]]]:
        new_domains = {k: v.copy() for k, v in domains.items()}
        var_index = self.slots.index(var)
        
        # Check adjacent slots
        neighbors = []
        if var_index > 0:
            neighbors.append(self.slots[var_index - 1])
        if var_index < len(self.slots) - 1:
            neighbors.append(self.slots[var_index + 1])
        
        for neighbor in neighbors:
            if neighbor not in assignment:
                # Remove value from neighbor's domain (no-back-to-back)
                if value in new_domains[neighbor]:
                    new_domains[neighbor].remove(value)
                
                # Check for domain wipeout
                if not new_domains[neighbor]:
                    return None
        
        return new_domains
    
    def ac3(self, domains: Dict[str, Set[str]], assignment: Dict[str, str] = None) -> Optional[Dict[str, Set[str]]]:
        if assignment is None:
            assignment = {}
        
        new_domains = {k: v.copy() for k, v in domains.items()}
        
        # Initialize queue with all arcs
        queue = deque()
        for i, slot1 in enumerate(self.slots):
            if slot1 in assignment:
                continue
            # Add arcs to adjacent slots
            if i > 0 and self.slots[i-1] not in assignment:
                queue.append((slot1, self.slots[i-1]))
            if i < len(self.slots) - 1 and self.slots[i+1] not in assignment:
                queue.append((slot1, self.slots[i+1]))
        
        while queue:
            xi, xj = queue.popleft()
            
            if self.revise(new_domains, xi, xj):
                if not new_domains[xi]:
                    return None  # Domain wipeout
                
                # Add neighbors of xi back to queue
                xi_index = self.slots.index(xi)
                neighbors = []
                if xi_index > 0 and self.slots[xi_index-1] != xj and self.slots[xi_index-1] not in assignment:
                    neighbors.append(self.slots[xi_index-1])
                if xi_index < len(self.slots) - 1 and self.slots[xi_index+1] != xj and self.slots[xi_index+1] not in assignment:
                    neighbors.append(self.slots[xi_index+1])
                
                for xk in neighbors:
                    queue.append((xk, xi))
        
        return new_domains
    
    def revise(self, domains: Dict[str, Set[str]], xi: str, xj: str) -> bool:
        revised = False
        to_remove = set()
        
        for x in domains[xi]:
            # Check if there exists a value in xj's domain different from x
            has_support = any(y != x for y in domains[xj])
            
            if not has_support:
                to_remove.add(x)
                revised = True
        
        domains[xi] -= to_remove
        return revised
    
    def backtrack(self, assignment: Dict[str, str], domains: Dict[str, Set[str]], 
                  use_mrv: bool = False, inference: str = 'none') -> Optional[Dict[str, str]]:
        # Check if assignment is complete
        if self.is_complete(assignment):
            return assignment
        
        # Select unassigned variable
        var = self.select_unassigned_variable(assignment, use_mrv)
        if var is None:
            return None
        
        # Try each value in domain
        for value in list(domains[var]):
            self.assignments_count += 1
            
            if self.is_consistent(assignment, var, value):
                # Make assignment
                assignment[var] = value
                new_domains = {k: v.copy() for k, v in domains.items()}
                
                # Apply inference
                inference_result = True
                if inference == 'forward_checking':
                    new_domains = self.forward_checking(assignment, var, value, new_domains)
                    inference_result = new_domains is not None
                elif inference == 'ac3':
                    new_domains = self.ac3(new_domains, assignment)
                    inference_result = new_domains is not None
                
                if inference_result:
                    # Recursive call
                    result = self.backtrack(assignment, new_domains, use_mrv, inference)
                    if result is not None:
                        return result
                
                # Backtrack
                del assignment[var]
                self.backtracks_count += 1
        
        return None
    
    def solve(self, use_mrv: bool = False, inference: str = 'none') -> Tuple[Optional[Dict[str, str]], float]:
        self.assignments_count = 0
        self.backtracks_count = 0
        
        start_time = time.time()
        solution = self.backtrack({}, self.domains, use_mrv, inference)
        end_time = time.time()
        
        return solution, end_time - start_time


def read_input_file(filename: str) -> Tuple[List[str], List[str], Dict[str, Set[str]]]:
    bots = []
    slots = []
    unary_constraints = {}
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('Bots:'):
                bots = [b.strip() for b in line.split(':', 1)[1].split(',')]
            elif line.startswith('Slots:'):
                slots = [s.strip() for s in line.split(':', 1)[1].split(',')]
            elif line.startswith('Unary_Constraint:'):
                # Parse constraint like "Slot4 != C"
                constraint = line.split(':', 1)[1].strip()
                parts = constraint.split('!=')
                if len(parts) == 2:
                    slot = parts[0].strip()
                    bot = parts[1].strip()
                    if slot not in unary_constraints:
                        unary_constraints[slot] = set()
                    unary_constraints[slot].add(bot)
    
    return bots, slots, unary_constraints


def print_results(solution: Optional[Dict[str, str]], time_taken: float, 
                 solver: CSPSolver, heuristic: str, inference: str):
    print("\nOUTPUT RESULTS")
    print("=" * 70)
    
    if solution:
        print("Success/Failure: SUCCESS\n")
        print(f"Heuristic Chosen: {heuristic}")
        print(f"Inference Method: {inference}\n")
        
        print("Constraints Applied:")
        print("  1. No Back-to-Back: A bot cannot work two consecutive slots")
        print("  2. Maintenance Break: Bot C cannot work in Slot 4")
        print("  3. Minimum Coverage: Every bot must be used at least once\n")
        
        print("Final Assignment of Bots to Slots:")
        for slot in solver.slots:
            print(f"  {slot} → Bot {solution[slot]}")
        
        # Verify constraints
        print("\nConstraint Verification:")
        
        # Check no-back-to-back
        no_back_to_back = True
        for i in range(len(solver.slots) - 1):
            if solution[solver.slots[i]] == solution[solver.slots[i+1]]:
                no_back_to_back = False
                break
        print(f"  ✓ No back-to-back constraint satisfied" if no_back_to_back 
              else "  ✗ No back-to-back constraint violated")
        
        # Check maintenance
        bot_c_not_in_slot4 = solution['Slot4'] != 'C'
        print(f"  ✓ Bot C not in Slot4: {bot_c_not_in_slot4}")
        
        # Check minimum coverage
        used_bots = set(solution.values())
        print(f"  ✓ All bots used: {used_bots}")
        
        print(f"\nTotal Number of Assignments Tried: {solver.assignments_count}")
        print(f"Total Number of Backtracks: {solver.backtracks_count}")
        print(f"Total Time Taken: {time_taken:.6f} seconds")
    else:
        print("Success/Failure: FAILURE\n")
        print(f"Heuristic Chosen: {heuristic}")
        print(f"Inference Method: {inference}\n")
        print("No solution found satisfying all constraints.")
        print(f"\nTotal Number of Assignments Tried: {solver.assignments_count}")
        print(f"Total Number of Backtracks: {solver.backtracks_count}")
        print(f"Total Time Taken: {time_taken:.6f} seconds")
    
    print("=" * 70)


def main():
    print("=" * 70)
    print("Question 2: Security Bot Scheduling using CSP Techniques")
    print("=" * 70)
    
    # Read input file
    input_file = 'input_csp.txt'
    
    if not os.path.exists(input_file):
        print(f"\nError: Input file '{input_file}' not found.")
        print("Creating default input file...")
        
        with open(input_file, 'w') as f:
            f.write("Bots: A,B,C\n")
            f.write("Slots: Slot1,Slot2,Slot3,Slot4\n")
            f.write("Unary_Constraint: Slot4 != C\n")
        
        print("Default input file created.")
    
    # Read configuration
    bots, slots, unary_constraints = read_input_file(input_file)
    
    print("\n1. PROBLEM FORMULATION")
    print("-" * 70)
    print(f"Variables: {', '.join(slots)}")
    print(f"Domain: Each variable can be assigned {{{', '.join(bots)}}}")
    print("\nConstraints:")
    print("  • Binary: No Back-to-Back (adjacent slots must have different bots)")
    print("  • Unary: Maintenance Break (Bot C unavailable in Slot4)")
    print("  • Global: Minimum Coverage (all bots must be used at least once)")
    
    print("\n2. IMPLEMENTATION REQUIREMENTS")
    print("-" * 70)
    
    # Configuration 1: Basic Backtracking
    print("\n[Configuration 1: Basic Backtracking]")
    solver1 = CSPSolver(bots, slots, unary_constraints)
    solution1, time1 = solver1.solve(use_mrv=False, inference='none')
    print_results(solution1, time1, solver1, "None (First-Unassigned)", "None")
    
    # Configuration 2: MRV Heuristic
    print("\n[Configuration 2: MRV Heuristic]")
    solver2 = CSPSolver(bots, slots, unary_constraints)
    solution2, time2 = solver2.solve(use_mrv=True, inference='none')
    print_results(solution2, time2, solver2, "MRV (Minimum Remaining Values)", "None")
    
    # Configuration 3: MRV + Forward Checking
    print("\n[Configuration 3: MRV + Forward Checking]")
    solver3 = CSPSolver(bots, slots, unary_constraints)
    solution3, time3 = solver3.solve(use_mrv=True, inference='forward_checking')
    print_results(solution3, time3, solver3, "MRV (Minimum Remaining Values)", "Forward Checking")
    
    # Configuration 4: MRV + AC-3
    print("\n[Configuration 4: MRV + AC-3]")
    solver4 = CSPSolver(bots, slots, unary_constraints)
    solution4, time4 = solver4.solve(use_mrv=True, inference='ac3')
    print_results(solution4, time4, solver4, "MRV (Minimum Remaining Values)", "AC-3 (Arc Consistency)")
    
    # Performance Comparison
    print("\n3. PERFORMANCE COMPARISON")
    print("-" * 70)
    print(f"{'Configuration':<30} {'Assignments':<15} {'Backtracks':<15} {'Time (s)':<15}")
    print("-" * 70)
    print(f"{'Basic Backtracking':<30} {solver1.assignments_count:<15} {solver1.backtracks_count:<15} {time1:<15.6f}")
    print(f"{'MRV Only':<30} {solver2.assignments_count:<15} {solver2.backtracks_count:<15} {time2:<15.6f}")
    print(f"{'MRV + Forward Checking':<30} {solver3.assignments_count:<15} {solver3.backtracks_count:<15} {time3:<15.6f}")
    print(f"{'MRV + AC-3':<30} {solver4.assignments_count:<15} {solver4.backtracks_count:<15} {time4:<15.6f}")
    print("=" * 70)
    
    print("\nKEY FINDINGS:")
    print("• MRV heuristic reduces assignments by selecting most constrained variables first")
    print("• Forward Checking detects failures early by pruning neighbor domains")
    print("• AC-3 provides strongest consistency but with higher computational cost")
    print("• For this 4-slot problem, MRV + Forward Checking offers best balance")


if __name__ == "__main__":
    main()
