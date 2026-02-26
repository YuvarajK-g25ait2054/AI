# Create a sample input.txt file
with open('input.txt', 'w') as f:
    f.write('Start: 1;2;3;4;B;5;6;7;8\n')
    f.write('Goal: 1;2;3;4;5;6;7;B;8\n')

print("input.txt created successfully with sample data.")


import time
import copy
from typing import Tuple, Optional, List, Set


class PuzzleState:
    def __init__(self, board: List[List], parent=None, action=None, cost=0):
        self.board = board
        self.parent = parent
        self.action = action
        self.cost = cost
        self.blank_pos = self._find_blank()
        
    def _find_blank(self) -> Tuple[int, int]:
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 'B':
                    return (i, j)
        return (0, 0)
    
    def get_state_tuple(self) -> tuple:
        return tuple(tuple(row) for row in self.board)
    
    def __eq__(self, other):
        return self.board == other.board
    
    def __hash__(self):
        return hash(self.get_state_tuple())
    
    def __lt__(self, other):
        return self.cost < other.cost
    
    def get_successors(self, visited_states: Set[tuple] = None) -> List['PuzzleState']:
        successors = []
        row, col = self.blank_pos
        
        if visited_states is None:
            visited_states = set()
        
        # Define moves
        moves = [
            (-1, 0, 'UP'),
            (1, 0, 'DOWN'),
            (0, -1, 'LEFT'),
            (0, 1, 'RIGHT')
        ]
        
        for dr, dc, action in moves:
            new_row, new_col = row + dr, col + dc
            
            # Check bounds
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                # Create new board
                new_board = [row[:] for row in self.board]
                new_board[row][col], new_board[new_row][new_col] = \
                    new_board[new_row][new_col], new_board[row][col]
                
                # Create successor state
                successor = PuzzleState(new_board, self, action, self.cost + 1)
                
                # STATE TRACKING: Skip if already visited
                successor_tuple = successor.get_state_tuple()
                if successor_tuple not in visited_states:
                    successors.append(successor)
        
        return successors
    
    def display(self):
        for row in self.board:
            print(' '.join(str(x) for x in row))


class Heuristics:
    @staticmethod
    def manhattan_distance(state: PuzzleState, goal_state: PuzzleState) -> int:
        # Create goal position mapping
        goal_positions = {}
        for i in range(3):
            for j in range(3):
                if goal_state.board[i][j] != 'B':
                    goal_positions[goal_state.board[i][j]] = (i, j)
        
        total_distance = 0
        for i in range(3):
            for j in range(3):
                tile = state.board[i][j]
                if tile != 'B' and tile in goal_positions:
                    goal_i, goal_j = goal_positions[tile]
                    total_distance += abs(i - goal_i) + abs(j - goal_j)
        
        return total_distance


class AdversarialSearch:
    def __init__(self, goal_state: PuzzleState, max_depth: int = 6):
        self.goal_state = goal_state
        self.max_depth = max_depth
        self.nodes_evaluated = 0
        self.pruned_count = 0
        self.visited_states = set()  # Track visited states
        self.states_skipped = 0  # Track states skipped due to revisiting
    
    def utility(self, state: PuzzleState) -> int:
        return -Heuristics.manhattan_distance(state, self.goal_state)
    
    def is_terminal(self, state: PuzzleState, depth: int) -> bool:
        return (state.get_state_tuple() == self.goal_state.get_state_tuple() or 
                depth >= self.max_depth)
    
    def minimax(self, state: PuzzleState, depth: int, is_max_player: bool) -> Tuple[int, Optional[str]]:
        self.nodes_evaluated += 1
        current_state_tuple = state.get_state_tuple()
        
        # Add current state to visited
        self.visited_states.add(current_state_tuple)
        
        # Terminal condition
        if self.is_terminal(state, depth):
            return self.utility(state), None
        
        # Get successors with state tracking (avoid revisiting)
        successors = state.get_successors(self.visited_states)
        
        # Count skipped states
        all_successors = state.get_successors(set())
        self.states_skipped += len(all_successors) - len(successors)
        
        if not successors:
            return self.utility(state), None
        
        if is_max_player:
            # MAX player: maximize value
            max_value = float('-inf')
            best_action = None
            
            for successor in successors:
                value, _ = self.minimax(successor, depth + 1, False)
                
                if value > max_value:
                    max_value = value
                    best_action = successor.action
            
            return max_value, best_action
        
        else:
            # MIN player: minimize value
            min_value = float('inf')
            best_action = None
            
            for successor in successors:
                value, _ = self.minimax(successor, depth + 1, True)
                
                if value < min_value:
                    min_value = value
                    best_action = successor.action
            
            return min_value, best_action
    
    def alpha_beta(self, state: PuzzleState, depth: int, alpha: float, beta: float, 
                   is_max_player: bool) -> Tuple[int, Optional[str]]:
        self.nodes_evaluated += 1
        current_state_tuple = state.get_state_tuple()
        
        # Add current state to visited
        self.visited_states.add(current_state_tuple)
        
        # Terminal condition
        if self.is_terminal(state, depth):
            return self.utility(state), None
        
        # Get successors with state tracking (avoid revisiting)
        successors = state.get_successors(self.visited_states)
        
        # Count skipped states
        all_successors = state.get_successors(set())
        self.states_skipped += len(all_successors) - len(successors)
        
        if not successors:
            return self.utility(state), None
        
        if is_max_player:
            # MAX player: maximize value
            max_value = float('-inf')
            best_action = None
            
            for successor in successors:
                value, _ = self.alpha_beta(successor, depth + 1, alpha, beta, False)
                
                if value > max_value:
                    max_value = value
                    best_action = successor.action
                
                alpha = max(alpha, value)
                
                # Beta cutoff: Prune
                if beta <= alpha:
                    self.pruned_count += 1
                    break
            
            return max_value, best_action
        
        else:
            # MIN player: minimize value
            min_value = float('inf')
            best_action = None
            
            for successor in successors:
                value, _ = self.alpha_beta(successor, depth + 1, alpha, beta, True)
                
                if value < min_value:
                    min_value = value
                    best_action = successor.action
                
                beta = min(beta, value)
                
                # Alpha cutoff: Prune
                if beta <= alpha:
                    self.pruned_count += 1
                    break
            
            return min_value, best_action
    
    def run_minimax(self, start: PuzzleState) -> dict:
        print("\n" + "="*70)
        print("MINIMAX ALGORITHM WITH STATE TRACKING")
        print("="*70)
        print("MAX Player: Robotic sorter (reach goal)")
        print("MIN Player: System glitch (increase disorder)")
        print(f"Max Depth: {self.max_depth}")
        print("Utility: Negative Manhattan distance")
        print("State Tracking: Visited/Not Visited lists maintained")
        print("Cycle Prevention: No revisiting explored states")
        print("="*70 + "\n")
        
        # Reset counters
        self.nodes_evaluated = 0
        self.visited_states = set()
        self.states_skipped = 0
        
        start_time = time.time()
        value, action = self.minimax(start, 0, True)
        end_time = time.time()
        
        total_time = end_time - start_time
        is_at_goal = start.get_state_tuple() == self.goal_state.get_state_tuple()
        success = is_at_goal or (action is not None)
        
        # Display results
        print("\n" + "="*70)
        print("OUTPUT RESULTS")
        print("="*70)
        print(f"Success/Failure: {'SUCCESS' if success else 'FAILURE'}")
        print(f"\nParameters Used:")
        print(f"  - Algorithm: Minimax with State Tracking")
        print(f"  - Max Depth: {self.max_depth}")
        print(f"  - Utility Function: Negative Manhattan Distance")
        print(f"  - Players: MAX (Sorter) vs MIN (Glitch)")
        print(f"\n(Sub)Optimal Path:")
        if is_at_goal:
            print(f"  Already at goal state")
        elif action:
            print(f"  Best Next Move: {action}")
            print(f"  Expected Utility: {value}")
        else:
            print(f"  No valid moves available")
        
        print(f"\n--- State Tracking & Performance ---")
        print(f"Total States Explored: {self.nodes_evaluated}")
        print(f"\nVisited List:")
        print(f"  Count: {len(self.visited_states)} states")
        print(f"  (All explored states maintained to prevent cycles)")
        print(f"\nNot Visited List (Filtered):")
        print(f"  Count: {self.states_skipped} states")
        print(f"  (States skipped due to already being visited)")
        print(f"\nConstraint Verification:")
        print(f"  ✓ Root state not revisited (added to visited list immediately)")
        print(f"  ✓ Parent states not revisited (already in visited list)")
        print(f"  ✓ Previously explored states not revisited (filtered out)")
        print(f"  ✓ Total states prevented from revisiting: {self.states_skipped}")
        print(f"\nEfficiency Gain: {(self.states_skipped/(self.nodes_evaluated + self.states_skipped)*100):.1f}% reduction in exploration")
        print(f"\nTotal Time Taken: {total_time:.6f} seconds")
        print("="*70)
        
        return {
            'algorithm': 'Minimax',
            'success': success,
            'best_value': value,
            'best_action': action,
            'nodes_evaluated': self.nodes_evaluated,
            'visited_states': len(self.visited_states),
            'states_skipped': self.states_skipped,
            'time': total_time
        }
    
    def run_alpha_beta(self, start: PuzzleState) -> dict:
        print("\n" + "="*70)
        print("ALPHA-BETA PRUNING WITH STATE TRACKING")
        print("="*70)
        print("MAX Player: Robotic sorter (reach goal)")
        print("MIN Player: System glitch (increase disorder)")
        print(f"Max Depth: {self.max_depth}")
        print("Utility: Negative Manhattan distance")
        print("Pruning: α (best for MAX), β (best for MIN)")
        print("State Tracking: Visited/Not Visited lists maintained")
        print("Cycle Prevention: No revisiting explored states")
        print("="*70 + "\n")
        
        # Reset counters
        self.nodes_evaluated = 0
        self.pruned_count = 0
        self.visited_states = set()
        self.states_skipped = 0
        
        start_time = time.time()
        value, action = self.alpha_beta(start, 0, float('-inf'), float('inf'), True)
        end_time = time.time()
        
        total_time = end_time - start_time
        is_at_goal = start.get_state_tuple() == self.goal_state.get_state_tuple()
        success = is_at_goal or (action is not None)
        
        # Display results
        print("\n" + "="*70)
        print("OUTPUT RESULTS")
        print("="*70)
        print(f"Success/Failure: {'SUCCESS' if success else 'FAILURE'}")
        print(f"\nParameters Used:")
        print(f"  - Algorithm: Alpha-Beta Pruning with State Tracking")
        print(f"  - Max Depth: {self.max_depth}")
        print(f"  - Utility Function: Negative Manhattan Distance")
        print(f"  - Pruning: α (MAX lower bound), β (MIN upper bound)")
        print(f"  - Players: MAX (Sorter) vs MIN (Glitch)")
        print(f"\n(Sub)Optimal Path:")
        if is_at_goal:
            print(f"  Already at goal state")
        elif action:
            print(f"  Best Next Move: {action}")
            print(f"  Expected Utility: {value}")
        else:
            print(f"  No valid moves available")
        
        print(f"\n--- State Tracking & Performance ---")
        print(f"Total States Explored: {self.nodes_evaluated}")
        print(f"\nVisited List:")
        print(f"  Count: {len(self.visited_states)} states")
        print(f"  (All explored states maintained to prevent cycles)")
        print(f"\nNot Visited List (Filtered):")
        print(f"  Count: {self.states_skipped} states")
        print(f"  (States skipped due to already being visited)")
        print(f"\nConstraint Verification:")
        print(f"  ✓ Root state not revisited (added to visited list immediately)")
        print(f"  ✓ Parent states not revisited (already in visited list)")
        print(f"  ✓ Previously explored states not revisited (filtered out)")
        print(f"  ✓ Total states prevented from revisiting: {self.states_skipped}")
        print(f"\nBranches Pruned (Alpha-Beta cutoff): {self.pruned_count}")
        total_potential_nodes = self.nodes_evaluated + self.states_skipped + self.pruned_count
        print(f"\nTotal Efficiency Gain: {((self.states_skipped + self.pruned_count)/max(1, total_potential_nodes)*100):.1f}% reduction in exploration")
        print(f"\nTotal Time Taken: {total_time:.6f} seconds")
        print("="*70)
        
        return {
            'algorithm': 'Alpha-Beta',
            'success': success,
            'best_value': value,
            'best_action': action,
            'nodes_evaluated': self.nodes_evaluated,
            'visited_states': len(self.visited_states),
            'states_skipped': self.states_skipped,
            'pruned_branches': self.pruned_count,
            'time': total_time
        }


def compare_algorithms(minimax_result: dict, alpha_beta_result: dict):
    print("\n" + "="*70)
    print("COMPARISON: MINIMAX vs ALPHA-BETA (with State Tracking)")
    print("="*70)
    
    print("\n1. OPTIMALITY VERIFICATION")
    print("-" * 70)
    print(f"Minimax Best Value:    {minimax_result['best_value']}")
    print(f"Alpha-Beta Best Value: {alpha_beta_result['best_value']}")
    print(f"Minimax Best Action:   {minimax_result['best_action']}")
    print(f"Alpha-Beta Best Action: {alpha_beta_result['best_action']}")
    
    if minimax_result['best_value'] == alpha_beta_result['best_value']:
        print("\n✓ Both algorithms found the SAME optimal value (Alpha-Beta is correct)")
    else:
        print("\n⚠ Different values found")
    
    print("\n2. EFFICIENCY COMPARISON")
    print("-" * 70)
    print(f"{'Metric':<35} {'Minimax':<15} {'Alpha-Beta':<15} {'Improvement'}")
    print("-" * 70)
    
    # States explored
    nodes_diff = minimax_result['nodes_evaluated'] - alpha_beta_result['nodes_evaluated']
    nodes_pct = (nodes_diff / minimax_result['nodes_evaluated'] * 100) if minimax_result['nodes_evaluated'] > 0 else 0
    print(f"{'States Explored':<35} {minimax_result['nodes_evaluated']:<15} {alpha_beta_result['nodes_evaluated']:<15} {nodes_pct:.1f}%")
    
    # Unique visited states
    print(f"{'Unique States Visited':<35} {minimax_result['visited_states']:<15} {alpha_beta_result['visited_states']:<15}")
    
    # States skipped
    print(f"{'States Skipped (Cycles)':<35} {minimax_result['states_skipped']:<15} {alpha_beta_result['states_skipped']:<15}")
    
    # Branches pruned (Alpha-Beta only)
    print(f"{'Branches Pruned (α-β cutoff)':<35} {'N/A':<15} {alpha_beta_result['pruned_branches']:<15}")
    
    # Time
    time_diff = minimax_result['time'] - alpha_beta_result['time']
    time_pct = (time_diff / minimax_result['time'] * 100) if minimax_result['time'] > 0 else 0
    print(f"{'Time Taken (seconds)':<35} {minimax_result['time']:.6f}{'s':<9} {alpha_beta_result['time']:.6f}{'s':<9} {time_pct:.1f}%")
    
    print("\n3. KEY INSIGHTS")
    print("-" * 70)
    print("✓ State Tracking prevents revisiting explored states (cycles)")
    print(f"✓ Alpha-Beta pruning reduced nodes by {nodes_pct:.1f}%")
    print(f"✓ Combined (State Tracking + α-β), avoided exploring {minimax_result['states_skipped'] + alpha_beta_result['pruned_branches']} states")
    print(f"✓ Time speedup: {time_pct:.1f}% faster with Alpha-Beta")
    print("=" * 70)


def read_input_file(filename: str = 'input.txt') -> Tuple[PuzzleState, PuzzleState]:
    """Read start and goal states from input file"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    start_board = None
    goal_board = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('Start:'):
            tiles = line.split(':', 1)[1].strip().split(';')
            start_board = [tiles[i:i+3] for i in range(0, 9, 3)]
        elif line.startswith('Goal:'):
            tiles = line.split(':', 1)[1].strip().split(';')
            goal_board = [tiles[i:i+3] for i in range(0, 9, 3)]
    
    return PuzzleState(start_board), PuzzleState(goal_board)


def get_manual_input() -> Tuple[PuzzleState, PuzzleState]:
    """Get start and goal states through manual input"""
    print("\n" + "="*70)
    print("MANUAL INPUT MODE - MATRIX FORMAT")
    print("="*70)
    print("Enter the puzzle state row by row (3 values per row).")
    print("Use 'B' for the blank tile. Separate values with spaces.")
    print("Example for a row: 1 2 3")
    print("-"*70)
    
    def input_state(state_name: str) -> PuzzleState:
        """Helper function to input a single state"""
        print(f"\n{state_name} State:")
        board = []
        tiles_list = []
        
        for row_num in range(1, 4):
            while True:
                try:
                    row_input = input(f"  Row {row_num} (3 values): ").strip()
                    values = row_input.split()
                    
                    if len(values) != 3:
                        print("    ❌ Error: Must enter exactly 3 values!")
                        continue
                    
                    board.append(values)
                    tiles_list.extend(values)
                    break
                except Exception as e:
                    print(f"    ❌ Error: {e}. Please try again.")
        
        # Validate complete board
        if 'B' not in tiles_list:
            print("❌ Error: Must include blank tile 'B'!")
            return None
        
        if len(tiles_list) != 9:
            print("❌ Error: Must have exactly 9 tiles total!")
            return None
        
        state = PuzzleState(board)
        print(f"\n✓ {state_name} state accepted:")
        state.display()
        return state
    
    # Get start state
    while True:
        start_state = input_state("START")
        if start_state is not None:
            break
        print("\nPlease try again.\n")
    
    # Get goal state
    while True:
        goal_state = input_state("GOAL")
        if goal_state is not None:
            break
        print("\nPlease try again.\n")
    
    return start_state, goal_state


def display_menu():
    """Display algorithm selection menu"""
    print("\n" + "="*70)
    print("ADVERSARIAL SEARCH ALGORITHM MENU")
    print("="*70)
    print("\n1. Minimax Algorithm")
    print("2. Alpha-Beta Pruning")
    print("3. Run Both & Compare")
    print("0. Exit")
    print("="*70)


def main():
    print("="*70)
    print("AI ASSIGNMENT - QUESTION 1: ADVERSARIAL SEARCH")
    print("State Tracking & Cycle Prevention Implementation")
    print("="*70)
    
    # Input mode selection
    print("\nINPUT MODE:")
    print("1. Read from input.txt file")
    print("2. Manual input")
    
    while True:
        mode = input("\nSelect input mode (1 or 2): ").strip()
        if mode in ['1', '2']:
            break
        print("❌ Invalid choice! Please enter 1 or 2.")
    
    # Get initial and goal states
    if mode == '1':
        try:
            start, goal = read_input_file('input.txt')
            print("\n✓ Successfully read from input.txt")
        except FileNotFoundError:
            print("\n⚠ input.txt not found. Creating default input.txt...")
            with open('input.txt', 'w') as f:
                f.write('Start: 1;2;3;4;B;5;6;7;8\n')
                f.write('Goal: 1;2;3;4;5;6;7;B;8\n')
            start, goal = read_input_file('input.txt')
            print("✓ Default input.txt created and loaded")
    else:
        start, goal = get_manual_input()
    
    # Get max depth parameter
    print("\n" + "="*70)
    print("ADVERSARIAL SEARCH PARAMETERS")
    print("="*70)
    while True:
        try:
            max_depth = int(input("\nEnter max search depth (recommended 4-8): ").strip())
            if max_depth > 0:
                break
            print("❌ Depth must be positive!")
        except ValueError:
            print("❌ Invalid input! Please enter a number.")
    
    print("\n" + "="*70)
    print("PUZZLE CONFIGURATION")
    print("="*70)
    print("\nInitial State:")
    start.display()
    print("\nGoal State:")
    goal.display()
    print(f"\nMax Search Depth: {max_depth}")
    
    # Algorithm selection loop
    while True:
        display_menu()
        choice = input("\nSelect option (0-3): ").strip()
        
        if choice == '0':
            print("\n✓ Exiting program. Thank you!")
            break
        
        if choice == '1':
            # Minimax only
            adversarial = AdversarialSearch(goal, max_depth)
            adversarial.run_minimax(start)
        
        elif choice == '2':
            # Alpha-Beta only
            adversarial = AdversarialSearch(goal, max_depth)
            adversarial.run_alpha_beta(start)
        
        elif choice == '3':
            # Run both and compare
            print("\n" + "="*70)
            print("RUNNING BOTH ALGORITHMS")
            print("="*70)
            
            # Minimax
            adversarial = AdversarialSearch(goal, max_depth)
            minimax_result = adversarial.run_minimax(start)
            
            # Alpha-Beta
            adversarial = AdversarialSearch(goal, max_depth)
            alpha_beta_result = adversarial.run_alpha_beta(start)
            
            # Compare
            compare_algorithms(minimax_result, alpha_beta_result)
        
        else:
            print("❌ Invalid choice! Please select 0-3.")
            continue
        
        # Ask to continue
        cont = input("\nRun another algorithm? (y/n): ").strip().lower()
        if cont != 'y':
            print("\n✓ Exiting program. Thank you!")
            break
    
    print("\n" + "="*70)
    print("STATE TRACKING FEATURES:")
    print("  ✓ Visited States List maintained")
    print("  ✓ Not Visited States filtered out")
    print("  ✓ No revisiting root, parent, or explored states")
    print("  ✓ Cycle prevention ensures efficient search")
    print("  ✓ Performance metrics show efficiency gains")
    print("="*70)


if __name__ == "__main__":
    main()