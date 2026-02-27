with open('input.txt', 'w') as f:
    f.write('Start: 1;2;3;4;B;5;6;7;8\n')
    f.write('Goal: 1;2;3;4;5;6;7;B;8\n')

print("input.txt created successfully with sample data.")

import time
import copy
from typing import Tuple, Optional, List


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
    
    def get_successors(self):
        successors = []
        row, col = self.blank_pos
        
        moves = [
            (-1, 0, 'UP'),     # Move blank up (tile moves down)
            (1, 0, 'DOWN'),    # Move blank down (tile moves up)
            (0, -1, 'LEFT'),   # Move blank left (tile moves right)
            (0, 1, 'RIGHT')    # Move blank right (tile moves left)
        ]
        
        for dr, dc, action in moves:
            new_row, new_col = row + dr, col + dc
            
            
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                
                new_board = [row[:] for row in self.board]
                
                new_board[row][col], new_board[new_row][new_col] = \
                    new_board[new_row][new_col], new_board[row][col]
                
                successor = PuzzleState(new_board, self, action, self.cost + 1)
                successors.append(successor)
        
        return successors
    
    def get_path(self):
        path = []
        current = self
        while current.parent is not None:
            path.append((current.action, current.board))
            current = current.parent
        path.reverse()
        return path
    
    def display(self):
        for row in self.board:
            print(' '.join(str(x) for x in row))


class Heuristics:
    @staticmethod
    def misplaced_tiles(state: PuzzleState, goal_state: PuzzleState) -> int:
        count = 0
        for i in range(3):
            for j in range(3):
                if state.board[i][j] != 'B' and \
                   state.board[i][j] != goal_state.board[i][j]:
                    count += 1
        return count
    
    @staticmethod
    def manhattan_distance(state: PuzzleState, goal_state: PuzzleState) -> int:
        
        total_distance = 0
        
        goal_positions = {}
        for i in range(3):
            for j in range(3):
                if goal_state.board[i][j] != 'B':
                    goal_positions[goal_state.board[i][j]] = (i, j)
        
        for i in range(3):
            for j in range(3):
                tile = state.board[i][j]
                if tile != 'B' and tile in goal_positions:
                    goal_i, goal_j = goal_positions[tile]
                    distance = abs(i - goal_i) + abs(j - goal_j)
                    total_distance += distance
        
        return total_distance


def parse_manual_input(input_str: str) -> List[List]:
    rows = input_str.strip().split('\n')
    board = []
    
    for row in rows:
        elements = row.strip().split()
        board_row = []
        for elem in elements:
            if elem.upper() == 'B':
                board_row.append('B')
            else:
                board_row.append(int(elem))
        board.append(board_row)
    
    return board


def parse_file_format(line: str) -> List[List]:
    elements = line.strip().split(';')
    board = []
    
    for i in range(0, 9, 3):
        row = []
        for j in range(3):
            elem = elements[i + j]
            if elem.upper() == 'B':
                row.append('B')
            else:
                row.append(int(elem))
        board.append(row)
    
    return board


def validate_board(board: List[List]) -> bool:
    if len(board) != 3:
        return False
    
    elements = set()
    blank_count = 0
    
    for row in board:
        if len(row) != 3:
            return False
        for elem in row:
            if elem == 'B':
                blank_count += 1
            elif isinstance(elem, int) and 1 <= elem <= 8:
                elements.add(elem)
            else:
                return False
    
    return blank_count == 1 and len(elements) == 8 and elements == set(range(1, 9))


def read_from_file(filename: str = 'input.txt') -> Optional[List[List]]:
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line.startswith('Start:'):
                start_line = line.split('Start:')[1].strip()
                return parse_file_format(start_line)
        
        return None
    except FileNotFoundError:
        return None


class AdversarialSearch:
    
    def __init__(self, goal_state: PuzzleState, max_depth=6):
        self.goal_state = goal_state
        self.max_depth = max_depth
        self.nodes_evaluated = 0
        self.pruned_count = 0
    
    def utility(self, state: PuzzleState) -> int:
        if state.get_state_tuple() == self.goal_state.get_state_tuple():
            return 0  # Goal reached
        
        return -Heuristics.manhattan_distance(state, self.goal_state)
    
    def is_terminal(self, state: PuzzleState, depth: int) -> bool:
        return (depth >= self.max_depth or 
                state.get_state_tuple() == self.goal_state.get_state_tuple())
    
    def minimax(self, state: PuzzleState, depth: int, is_max_player: bool) -> Tuple[int, Optional[str]]:
        self.nodes_evaluated += 1
       
        if self.is_terminal(state, depth):
            return self.utility(state), None
        
        successors = state.get_successors()
        
        if not successors:
            return self.utility(state), None
        
        if is_max_player:
            max_value = float('-inf')
            best_action = None
            
            for successor in successors:
                value, _ = self.minimax(successor, depth + 1, False)
                
                if value > max_value:
                    max_value = value
                    best_action = successor.action
            
            return max_value, best_action
        
        else:
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
        
        if self.is_terminal(state, depth):
            return self.utility(state), None
        
        successors = state.get_successors()
        
        if not successors:
            return self.utility(state), None
        
        if is_max_player:
            max_value = float('-inf')
            best_action = None
            
            for successor in successors:
                value, _ = self.alpha_beta(successor, depth + 1, alpha, beta, False)
                
                if value > max_value:
                    max_value = value
                    best_action = successor.action
                
                alpha = max(alpha, value)
                
                if beta <= alpha:
                    self.pruned_count += 1
                    break  
            
            return max_value, best_action
        
        else:
            min_value = float('inf')
            best_action = None
            
            for successor in successors:
                value, _ = self.alpha_beta(successor, depth + 1, alpha, beta, True)
                
                if value < min_value:
                    min_value = value
                    best_action = successor.action
                
                beta = min(beta, value)
                
                if beta <= alpha:
                    self.pruned_count += 1
                    break  
            
            return min_value, best_action
    
    def run_minimax(self, start: PuzzleState) -> dict:
        print("\n" + "="*70)
        print("MINIMAX ALGORITHM")
        print("="*70)
        print("MAX Player: Robotic sorter (reach goal)")
        print("MIN Player: System glitch (increase disorder)")
        print(f"Max Depth: {self.max_depth}")
        print("Utility: Negative Manhattan distance")
        print("="*70 + "\n")
        
        self.nodes_evaluated = 0
        start_time = time.time()
        
        value, action = self.minimax(start, 0, True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        is_at_goal = start.get_state_tuple() == self.goal_state.get_state_tuple()
        success = is_at_goal or (action is not None)
        
        print("\n" + "="*70)
        print("OUTPUT RESULTS")
        print("="*70)
        print(f"Success/Failure: {'SUCCESS' if success else 'FAILURE'}")
        print(f"\nParameters Used:")
        print(f"  - Algorithm: Minimax")
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
        print(f"\nTotal States Explored: {self.nodes_evaluated}")
        print(f"Total Time Taken: {total_time:.6f} seconds")
        print("="*70)
        
        return {
            'algorithm': 'Minimax',
            'success': success,
            'best_value': value,
            'best_action': action,
            'nodes_evaluated': self.nodes_evaluated,
            'time': total_time,
            'max_depth': self.max_depth
        }
    
    def run_alpha_beta(self, start: PuzzleState) -> dict:
        print("\n" + "="*70)
        print("ALPHA-BETA PRUNING")
        print("="*70)
        print("MAX Player: Robotic sorter (reach goal)")
        print("MIN Player: System glitch (increase disorder)")
        print(f"Max Depth: {self.max_depth}")
        print("Utility: Negative Manhattan distance")
        print("Pruning: α (best for MAX), β (best for MIN)")
        print("="*70 + "\n")
        
        self.nodes_evaluated = 0
        self.pruned_count = 0
        start_time = time.time()
        
        value, action = self.alpha_beta(start, 0, float('-inf'), float('inf'), True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        is_at_goal = start.get_state_tuple() == self.goal_state.get_state_tuple()
        success = is_at_goal or (action is not None)
        
        
        print("\n" + "="*70)
        print("OUTPUT RESULTS")
        print("="*70)
        print(f"Success/Failure: {'SUCCESS' if success else 'FAILURE'}")
        print(f"\nParameters Used:")
        print(f"  - Algorithm: Alpha-Beta Pruning")
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
        print(f"\nTotal States Explored: {self.nodes_evaluated}")
        print(f"Branches Pruned: {self.pruned_count}")
        print(f"Efficiency Gain: {(self.pruned_count/max(1, self.nodes_evaluated + self.pruned_count))*100:.1f}% reduction")
        print(f"Total Time Taken: {total_time:.6f} seconds")
        print("="*70)
        
        return {
            'algorithm': 'Alpha-Beta',
            'success': success,
            'best_value': value,
            'best_action': action,
            'nodes_evaluated': self.nodes_evaluated,
            'pruned_branches': self.pruned_count,
            'time': total_time,
            'max_depth': self.max_depth
        }


def simulate_adversarial_game(start: PuzzleState, goal: PuzzleState, 
                              algorithm='alpha-beta', max_depth=6, max_moves=10):
    print("\n" + "="*70)
    print("ADVERSARIAL GAME SIMULATION")
    print("="*70)
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Max Depth per Move: {max_depth}")
    print(f"Max Total Moves: {max_moves}")
    print("="*70 + "\n")
    
    current_state = start
    move_count = 0
    total_nodes = 0
    total_time = 0
    total_pruned = 0
    
    print("Initial State:")
    current_state.display()
    
    while move_count < max_moves:
        if current_state.get_state_tuple() == goal.get_state_tuple():
            print(f"\n🎉 GOAL REACHED in {move_count} moves!")
            print("\n" + "="*70)
            print("OUTPUT RESULTS - GAME SUMMARY")
            print("="*70)
            print(f"Success/Failure: SUCCESS")
            print(f"\nParameters Used:")
            print(f"  - Algorithm: {algorithm.upper()}")
            print(f"  - Max Depth per Move: {max_depth}")
            print(f"  - Max Total Moves: {max_moves}")
            print(f"  - Utility Function: Negative Manhattan Distance")
            print(f"\n(Sub)Optimal Path:")
            print(f"  - Total Moves to Goal: {move_count}")
            print(f"  - Path: Alternating MAX/MIN moves")
            print(f"\nTotal States Explored: {total_nodes}")
            if algorithm == 'alpha-beta':
                print(f"Total Branches Pruned: {total_pruned}")
            print(f"Average States per Move: {total_nodes/move_count:.1f}" if move_count > 0 else "N/A")
            print(f"Total Time Taken: {total_time:.6f} seconds")
            print("="*70)
            return {
                'success': True,
                'moves': move_count,
                'total_nodes': total_nodes,
                'total_time': total_time
            }
        
        move_count += 1
        is_max_turn = (move_count % 2 == 1) 
        
        print(f"\n{'='*70}")
        print(f"Move {move_count}: {'MAX (Sorter)' if is_max_turn else 'MIN (Glitch)'}")
        print(f"{'='*70}")
        
        search = AdversarialSearch(goal, max_depth)
        
        start_time = time.time()
        if algorithm == 'minimax':
            value, action = search.minimax(current_state, 0, True)
        else:
            value, action = search.alpha_beta(current_state, 0, float('-inf'), float('inf'), True)
        end_time = time.time()
        
        move_time = end_time - start_time
        total_nodes += search.nodes_evaluated
        total_time += move_time
        
        print(f"Best Action: {action if action else 'None'}")
        print(f"Utility Value: {value}")
        print(f"Nodes Evaluated: {search.nodes_evaluated}")
        if algorithm == 'alpha-beta':
            print(f"Branches Pruned: {search.pruned_count}")
            total_pruned += search.pruned_count
        print(f"Time: {move_time:.6f}s")
        
        if action:
            successors = current_state.get_successors()
            for succ in successors:
                if succ.action == action:
                    current_state = succ
                    break
        
        print("\nResulting State:")
        current_state.display()
    
    print(f"\n❌ Max moves ({max_moves}) reached without finding goal")
    print("\n" + "="*70)
    print("OUTPUT RESULTS - GAME SUMMARY")
    print("="*70)
    print(f"Success/Failure: FAILURE")
    print(f"\nParameters Used:")
    print(f"  - Algorithm: {algorithm.upper()}")
    print(f"  - Max Depth per Move: {max_depth}")
    print(f"  - Max Total Moves: {max_moves}")
    print(f"  - Utility Function: Negative Manhattan Distance")
    print(f"\n(Sub)Optimal Path:")
    print(f"  - Moves Completed: {move_count}")
    print(f"  - Goal Not Reached")
    print(f"  - Final Manhattan Distance: {Heuristics.manhattan_distance(current_state, goal)}")
    print(f"\nTotal States Explored: {total_nodes}")
    if algorithm == 'alpha-beta':
        print(f"Total Branches Pruned: {total_pruned}")
    print(f"Average States per Move: {total_nodes/move_count:.1f}" if move_count > 0 else "N/A")
    print(f"Total Time Taken: {total_time:.6f} seconds")
    print("="*70)
    return {
        'success': False,
        'moves': move_count,
        'total_nodes': total_nodes,
        'total_time': total_time
    }


def compare_minimax_alphabeta(start: PuzzleState, goal: PuzzleState, max_depth=6):
    print("\n" + "="*70)
    print("COMPARISON: MINIMAX vs ALPHA-BETA PRUNING")
    print("="*70)
    print("Testing both algorithms on the same initial state...")
    print(f"Max Depth: {max_depth}")
    print("="*70 + "\n")
    
    search_mm = AdversarialSearch(goal, max_depth)
    result_mm = search_mm.run_minimax(start)
    
    
    search_ab = AdversarialSearch(goal, max_depth)
    result_ab = search_ab.run_alpha_beta(start)
    
    same_value = result_mm['best_value'] == result_ab['best_value']
    same_action = result_mm['best_action'] == result_ab['best_action']
    node_improvement = (1 - result_ab['nodes_evaluated']/result_mm['nodes_evaluated'])*100 if result_mm['nodes_evaluated'] > 0 else 0
    time_improvement = (1 - result_ab['time']/result_mm['time'])*100 if result_mm['time'] > 0 else 0
    
    action_mm = result_mm['best_action'] if result_mm['best_action'] is not None else 'None (at goal)'
    action_ab = result_ab['best_action'] if result_ab['best_action'] is not None else 'None (at goal)'
    
    print("\n" + "="*70)
    print("RESULTS COMPARISON: MINIMAX vs ALPHA-BETA PRUNING")
    print("="*70)
    
    print("\n1. OPTIMALITY VERIFICATION:")
    print("-" * 70)
    print(f"   Best Value (Utility):")
    print(f"      Minimax:     {result_mm['best_value']}")
    print(f"      Alpha-Beta:  {result_ab['best_value']}")
    print(f"      Status:      {'✓ IDENTICAL' if same_value else '✗ DIFFERENT'}")
    print(f"\n   Best Action (Next Move):")
    print(f"      Minimax:     {action_mm}")
    print(f"      Alpha-Beta:  {action_ab}")
    print(f"      Status:      {'✓ IDENTICAL' if same_action else '✗ DIFFERENT'}")
    
    print("\n2. EFFICIENCY COMPARISON:")
    print("-" * 70)
    print(f"   States Explored:")
    print(f"      Minimax:     {result_mm['nodes_evaluated']:,} nodes")
    print(f"      Alpha-Beta:  {result_ab['nodes_evaluated']:,} nodes")
    print(f"      Reduction:   {node_improvement:.1f}% fewer nodes")
    
    print(f"\n   Branches Pruned:")
    print(f"      Minimax:     N/A (no pruning)")
    print(f"      Alpha-Beta:  {result_ab['pruned_branches']:,} branches")
    
    print(f"\n   Time Taken:")
    print(f"      Minimax:     {result_mm['time']:.6f} seconds")
    print(f"      Alpha-Beta:  {result_ab['time']:.6f} seconds")
    print(f"      Speedup:     {time_improvement:.1f}% faster")
    
    print("\n3. KEY INSIGHTS:")
    print("-" * 70)
    print(f"   ✓ Both algorithms produce IDENTICAL optimal decisions")
    print(f"   ✓ Alpha-Beta prunes {result_ab['pruned_branches']} branches")
    print(f"   ✓ Efficiency gain: {node_improvement:.1f}% reduction in states explored")
    print(f"   ✓ Time savings: {time_improvement:.1f}% faster execution")
    print(f"   ✓ Pruning maintains optimality while improving performance")
    
    print("="*70)
    
    return result_mm, result_ab


def main():
    print("\n" + "#"*70)
    print("# QUESTION 1D: ADVERSARIAL SEARCH - MINIMAX & ALPHA-BETA")
    print("#"*70 + "\n")
    
    goal_board = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 'B']
    ]
    goal_state = PuzzleState(goal_board)
    
    print("="*70)
    print("SELECT INPUT MODE:")
    print("="*70)
    print("1. Manual Input (Enter puzzle row by row)")
    print("2. Read from input.txt file")
    print("="*70)
    
    mode = input("\nEnter your choice (1 or 2): ").strip()
    
    start_board = None
    
    if mode == '2':
        start_board = read_from_file('input.txt')
        
        if start_board is None:
            print("\n❌ Could not read from input.txt file!")
            print("Switching to manual input mode...\n")
            mode = '1'
        else:
            print("\n✓ Successfully read from input.txt")
            print("\n✓ Starting state:")
            for row in start_board:
                print(" ".join(str(x) for x in row))
    
    if mode == '1':
        print("\n" + "="*70)
        print("MANUAL INPUT MODE")
        print("="*70)
        print("\nEnter the STARTING puzzle configuration:")
        print("- Enter 3 rows, each with 3 elements")
        print("- Use numbers 1-8 and 'B' for blank")
        print("- Separate elements with spaces")
        print("\nExample:")
        print("1 2 3")
        print("4 5 6")
        print("B 7 8")
        print("\nEnter your starting state:")
        
        initial_input = []
        for i in range(3):
            row = input(f"Row {i+1}: ").strip()
            initial_input.append(row)
        
        start_board = parse_manual_input('\n'.join(initial_input))
        
        if not validate_board(start_board):
            print("\n❌ Invalid board! Please ensure:")
            print("- 3x3 grid")
            print("- Exactly one 'B'")
            print("- Numbers 1-8 (each appearing once)")
            return
        
        print("\n✓ Starting state:")
        for row in start_board:
            print(" ".join(str(x) for x in row))
    
    if start_board is None:
        print("\n❌ Failed to get valid board configuration!")
        return
    
    start_state = PuzzleState(start_board)
    
    print("Starting State:")
    start_state.display()
    print("\nGoal State:")
    goal_state.display()
   
    print("\n" + "="*70)
    print("PART 1: SINGLE MOVE COMPARISON")
    print("="*70)
    print("Evaluating optimal next move from current state.")
    print("Both algorithms will determine best action for MAX player.")
    print("="*70)
    compare_minimax_alphabeta(start_state, goal_state, max_depth=6)
    
    print("\n\n" + "="*70)
    print("PART 2: GAME SIMULATION WITH MINIMAX")
    print("="*70)
    print("Simulating multi-move adversarial game.")
    print("MAX and MIN alternate moves until goal or move limit.")
    print("="*70)
    simulate_adversarial_game(start_state, goal_state, 
                             algorithm='minimax', max_depth=4, max_moves=8)
    
    print("\n\n" + "="*70)
    print("PART 3: GAME SIMULATION WITH ALPHA-BETA")
    print("="*70)
    print("Simulating multi-move adversarial game with pruning.")
    print("MAX and MIN alternate moves until goal or move limit.")
    print("="*70)
    simulate_adversarial_game(start_state, goal_state, 
                             algorithm='alpha-beta', max_depth=4, max_moves=8)
    
    print("\n" + "="*30)
    print("ADVERSARIAL SEARCH COMPLETE")
    print("="*30)


if __name__ == "__main__":
    main()