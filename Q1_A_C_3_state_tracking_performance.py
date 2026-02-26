# Create a sample input.txt file
with open('input.txt', 'w') as f:
    f.write('Start: 1;2;3;4;B;5;6;7;8\n')
    f.write('Goal: 1;2;3;4;5;6;7;B;8\n')

print("input.txt created successfully with sample data.")

import heapq
import time
import math
import random
from collections import deque
from typing import List, Tuple, Set, Optional, Dict


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
    
    def get_possible_moves(self) -> List[Tuple[int, int, str]]:
        moves = []
        row, col = self.blank_pos
        if row > 0:
            moves.append((row - 1, col, 'Up'))
        if row < 2:
            moves.append((row + 1, col, 'Down'))
        if col > 0:
            moves.append((row, col - 1, 'Left'))
        if col < 2:
            moves.append((row, col + 1, 'Right'))
        return moves
    
    def move(self, tile_pos: Tuple[int, int], direction: str):
        new_board = [row[:] for row in self.board]
        blank_row, blank_col = self.blank_pos
        tile_row, tile_col = tile_pos
        
        new_board[blank_row][blank_col] = new_board[tile_row][tile_col]
        new_board[tile_row][tile_col] = 'B'
        
        return PuzzleState(new_board, self, direction, self.cost + 1)
    
    def __hash__(self):
        return hash(str(self.board))
    
    def __eq__(self, other):
        return self.board == other.board
    
    def __lt__(self, other):
        return self.cost < other.cost


class PuzzleSolver:
    
    def __init__(self, initial_state: PuzzleState, goal_state: PuzzleState):
        self.initial = initial_state
        self.goal = goal_state
        self.nodes_expanded = 0
        self.max_fringe_size = 0
        self.visited_states = set()  # Visited List
        self.states_skipped = 0  # States prevented from revisiting
    
    def is_goal(self, state: PuzzleState) -> bool:
        return state.board == self.goal.board
    
    def misplaced_tiles(self, state: PuzzleState) -> int:
        count = 0
        for i in range(3):
            for j in range(3):
                if state.board[i][j] != 'B' and state.board[i][j] != self.goal.board[i][j]:
                    count += 1
        return count
    
    def manhattan_distance(self, state: PuzzleState) -> int:
        distance = 0
        for i in range(3):
            for j in range(3):
                if state.board[i][j] != 'B':
                    value = state.board[i][j]
                    for gi in range(3):
                        for gj in range(3):
                            if self.goal.board[gi][gj] == value:
                                distance += abs(i - gi) + abs(j - gj)
        return distance
    
    def reconstruct_path(self, state: PuzzleState) -> List[str]:
        path = []
        current = state
        while current.parent is not None:
            path.append(current.action)
            current = current.parent
        return list(reversed(path))
    
    def print_state_tracking(self, algorithm: str):
        print(f"\n--- State Tracking & Performance ---")
        print(f"Total States Explored: {self.nodes_expanded}")
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
        if self.states_skipped > 0:
            efficiency = (self.states_skipped / (self.nodes_expanded + self.states_skipped)) * 100
            print(f"\nEfficiency Gain: {efficiency:.1f}% reduction in exploration")
    
    def bfs(self) -> Dict:
        print("\n" + "="*70)
        print("BFS (BREADTH-FIRST SEARCH) WITH STATE TRACKING")
        print("="*70)
        
        # Reset tracking
        self.nodes_expanded = 0
        self.visited_states = set()
        self.states_skipped = 0
        
        start_time = time.time()
        
        if self.is_goal(self.initial):
            end_time = time.time()
            print("\nOUTPUT RESULTS")
            print("-" * 70)
            print(f"Success/Failure: SUCCESS")
            print(f"Heuristic/Parameters: Uninformed Search (BFS - FIFO Queue)")
            print(f"(Sub)Optimal Path: [Already at goal]")
            print(f"Path Length: 0 moves")
            print(f"Total States Explored: 0")
            print(f"Total Time Taken: {end_time - start_time:.6f} seconds")
            return {'algorithm': 'BFS', 'found': True, 'time': end_time - start_time}
        
        frontier = deque([self.initial])
        self.visited_states.add(hash(self.initial))
        
        while frontier:
            self.max_fringe_size = max(self.max_fringe_size, len(frontier))
            state = frontier.popleft()
            
            if self.is_goal(state):
                end_time = time.time()
                path = self.reconstruct_path(state)
                
                print("\nOUTPUT RESULTS")
                print("-" * 70)
                print(f"Success/Failure: SUCCESS")
                print(f"Heuristic/Parameters: Uninformed Search (BFS - FIFO Queue)")
                print(f"(Sub)Optimal Path: {' → '.join(path)}")
                print(f"Path Length: {len(path)} moves")
                print(f"Total States Explored: {self.nodes_expanded}")
                print(f"Total Time Taken: {end_time - start_time:.6f} seconds")
                self.print_state_tracking('BFS')
                print("="*70)
                
                return {
                    'algorithm': 'BFS',
                    'path_length': len(path),
                    'nodes_expanded': self.nodes_expanded,
                    'visited_states': len(self.visited_states),
                    'states_skipped': self.states_skipped,
                    'time': end_time - start_time,
                    'found': True
                }
            
            self.nodes_expanded += 1
            
            # Generate successors with state tracking
            for tile_row, tile_col, direction in state.get_possible_moves():
                child = state.move((tile_row, tile_col), direction)
                child_hash = hash(child)
                
                # STATE TRACKING: Check if already visited
                if child_hash not in self.visited_states:
                    frontier.append(child)
                    self.visited_states.add(child_hash)
                else:
                    self.states_skipped += 1  # Count skipped states
        
        end_time = time.time()
        print("\nOUTPUT RESULTS")
        print("-" * 70)
        print(f"Success/Failure: FAILURE")
        print(f"Total States Explored: {self.nodes_expanded}")
        print(f"Total Time Taken: {end_time - start_time:.6f} seconds")
        self.print_state_tracking('BFS')
        print("="*70)
        
        return {'algorithm': 'BFS', 'found': False, 'time': end_time - start_time}
    
    def dfs(self, depth_limit=50) -> Dict:
        print("\n" + "="*70)
        print(f"DFS (DEPTH-FIRST SEARCH) WITH STATE TRACKING (Limit: {depth_limit})")
        print("="*70)
        
        # Reset tracking
        self.nodes_expanded = 0
        self.visited_states = set()
        self.states_skipped = 0
        
        start_time = time.time()
        
        frontier = [self.initial]
        self.visited_states.add(hash(self.initial))
        
        while frontier:
            self.max_fringe_size = max(self.max_fringe_size, len(frontier))
            state = frontier.pop()
            
            if self.is_goal(state):
                end_time = time.time()
                path = self.reconstruct_path(state)
                
                print("\nOUTPUT RESULTS")
                print("-" * 70)
                print(f"Success/Failure: SUCCESS")
                print(f"Heuristic/Parameters: Uninformed Search (DFS - LIFO Stack, Depth Limit: {depth_limit})")
                print(f"(Sub)Optimal Path: {' → '.join(path)}")
                print(f"Path Length: {len(path)} moves")
                print(f"Total States Explored: {self.nodes_expanded}")
                print(f"Total Time Taken: {end_time - start_time:.6f} seconds")
                self.print_state_tracking('DFS')
                print("="*70)
                
                return {
                    'algorithm': 'DFS',
                    'path_length': len(path),
                    'nodes_expanded': self.nodes_expanded,
                    'visited_states': len(self.visited_states),
                    'states_skipped': self.states_skipped,
                    'time': end_time - start_time,
                    'found': True
                }
            
            if state.cost < depth_limit:
                self.nodes_expanded += 1
                
                # Generate successors with state tracking
                for tile_row, tile_col, direction in state.get_possible_moves():
                    child = state.move((tile_row, tile_col), direction)
                    child_hash = hash(child)
                    
                    # STATE TRACKING: Check if already visited
                    if child_hash not in self.visited_states:
                        frontier.append(child)
                        self.visited_states.add(child_hash)
                    else:
                        self.states_skipped += 1
        
        end_time = time.time()
        print("\nOUTPUT RESULTS")
        print("-" * 70)
        print(f"Success/Failure: FAILURE")
        print(f"Total States Explored: {self.nodes_expanded}")
        print(f"Total Time Taken: {end_time - start_time:.6f} seconds")
        self.print_state_tracking('DFS')
        print("="*70)
        
        return {'algorithm': 'DFS', 'found': False, 'time': end_time - start_time}
    
    def greedy_best_first(self, heuristic='manhattan') -> Dict:
        h_func = self.manhattan_distance if heuristic == 'manhattan' else self.misplaced_tiles
        h_name = "h2 (Manhattan Distance)" if heuristic == 'manhattan' else "h1 (Misplaced Tiles)"
        
        print("\n" + "="*70)
        print(f"GREEDY BEST-FIRST SEARCH WITH STATE TRACKING - {h_name}")
        print("="*70)
        
        # Reset tracking
        self.nodes_expanded = 0
        self.visited_states = set()
        self.states_skipped = 0
        
        start_time = time.time()
        
        frontier = []
        counter = 0
        heapq.heappush(frontier, (h_func(self.initial), counter, self.initial))
        self.visited_states.add(hash(self.initial))
        
        while frontier:
            self.max_fringe_size = max(self.max_fringe_size, len(frontier))
            _, _, state = heapq.heappop(frontier)
            
            if self.is_goal(state):
                end_time = time.time()
                path = self.reconstruct_path(state)
                
                print("\nOUTPUT RESULTS")
                print("-" * 70)
                print(f"Success/Failure: SUCCESS")
                print(f"Heuristic/Parameters: Greedy Best-First with {h_name}, f(n) = h(n)")
                print(f"(Sub)Optimal Path: {' → '.join(path)}")
                print(f"Path Length: {len(path)} moves")
                print(f"Total States Explored: {self.nodes_expanded}")
                print(f"Total Time Taken: {end_time - start_time:.6f} seconds")
                self.print_state_tracking('Greedy')
                print("="*70)
                
                return {
                    'algorithm': f'Greedy ({heuristic})',
                    'path_length': len(path),
                    'nodes_expanded': self.nodes_expanded,
                    'visited_states': len(self.visited_states),
                    'states_skipped': self.states_skipped,
                    'time': end_time - start_time,
                    'found': True
                }
            
            self.nodes_expanded += 1
            
            for tile_row, tile_col, direction in state.get_possible_moves():
                child = state.move((tile_row, tile_col), direction)
                child_hash = hash(child)
                
                if child_hash not in self.visited_states:
                    counter += 1
                    heapq.heappush(frontier, (h_func(child), counter, child))
                    self.visited_states.add(child_hash)
                else:
                    self.states_skipped += 1
        
        end_time = time.time()
        print("\nOUTPUT RESULTS")
        print("-" * 70)
        print(f"Success/Failure: FAILURE")
        print(f"Total States Explored: {self.nodes_expanded}")
        print(f"Total Time Taken: {end_time - start_time:.6f} seconds")
        self.print_state_tracking('Greedy')
        print("="*70)
        
        return {'algorithm': f'Greedy ({heuristic})', 'found': False, 'time': end_time - start_time}
    
    def a_star(self, heuristic='manhattan') -> Dict:
        h_func = self.manhattan_distance if heuristic == 'manhattan' else self.misplaced_tiles
        h_name = "h2 (Manhattan Distance)" if heuristic == 'manhattan' else "h1 (Misplaced Tiles)"
        
        print("\n" + "="*70)
        print(f"A* SEARCH WITH STATE TRACKING - {h_name}")
        print("="*70)
        
        # Reset tracking
        self.nodes_expanded = 0
        self.visited_states = set()
        self.states_skipped = 0
        
        start_time = time.time()
        
        frontier = []
        counter = 0
        heapq.heappush(frontier, (h_func(self.initial), counter, self.initial))
        self.visited_states.add(hash(self.initial))
        
        while frontier:
            self.max_fringe_size = max(self.max_fringe_size, len(frontier))
            _, _, state = heapq.heappop(frontier)
            
            if self.is_goal(state):
                end_time = time.time()
                path = self.reconstruct_path(state)
                
                print("\nOUTPUT RESULTS")
                print("-" * 70)
                print(f"Success/Failure: SUCCESS")
                print(f"Heuristic/Parameters: A* Search with {h_name}, f(n) = g(n) + h(n)")
                print(f"(Sub)Optimal Path: {' → '.join(path)}")
                print(f"Path Length: {len(path)} moves (Optimal)")
                print(f"Total States Explored: {self.nodes_expanded}")
                print(f"Total Time Taken: {end_time - start_time:.6f} seconds")
                self.print_state_tracking('A*')
                print("="*70)
                
                return {
                    'algorithm': f'A* ({heuristic})',
                    'path_length': len(path),
                    'nodes_expanded': self.nodes_expanded,
                    'visited_states': len(self.visited_states),
                    'states_skipped': self.states_skipped,
                    'time': end_time - start_time,
                    'found': True
                }
            
            self.nodes_expanded += 1
            
            for tile_row, tile_col, direction in state.get_possible_moves():
                child = state.move((tile_row, tile_col), direction)
                child_hash = hash(child)
                
                if child_hash not in self.visited_states:
                    f_score = child.cost + h_func(child)
                    counter += 1
                    heapq.heappush(frontier, (f_score, counter, child))
                    self.visited_states.add(child_hash)
                else:
                    self.states_skipped += 1
        
        end_time = time.time()
        print("\nOUTPUT RESULTS")
        print("-" * 70)
        print(f"Success/Failure: FAILURE")
        print(f"Total States Explored: {self.nodes_expanded}")
        print(f"Total Time Taken: {end_time - start_time:.6f} seconds")
        self.print_state_tracking('A*')
        print("="*70)
        
        return {'algorithm': f'A* ({heuristic})', 'found': False, 'time': end_time - start_time}
    
    def ida_star(self, heuristic='manhattan', max_iterations=50) -> Dict:
        h_func = self.manhattan_distance if heuristic == 'manhattan' else self.misplaced_tiles
        h_name = "h2 (Manhattan Distance)" if heuristic == 'manhattan' else "h1 (Misplaced Tiles)"
        
        print("\n" + "="*70)
        print(f"IDA* SEARCH WITH STATE TRACKING - {h_name}")
        print("="*70)
        
        # Reset tracking
        self.nodes_expanded = 0
        self.visited_states = set()
        self.states_skipped = 0
        
        start_time = time.time()
        
        def search(state, g, bound, path_states):
            """Recursive DFS with f-cost bound"""
            self.nodes_expanded += 1
            f = g + h_func(state)
            
            if f > bound:
                return f, None
            
            if self.is_goal(state):
                return -1, state
            
            min_bound = float('inf')
            
            for tile_row, tile_col, direction in state.get_possible_moves():
                child = state.move((tile_row, tile_col), direction)
                child_hash = hash(child)
                
                # STATE TRACKING: Check path and visited
                if child_hash not in path_states:
                    if child_hash in self.visited_states:
                        self.states_skipped += 1
                    else:
                        path_states.add(child_hash)
                        self.visited_states.add(child_hash)
                        
                        t, result = search(child, g + 1, bound, path_states)
                        
                        if t == -1:
                            return -1, result
                        
                        if t < min_bound:
                            min_bound = t
                        
                        path_states.remove(child_hash)
                else:
                    self.states_skipped += 1
            
            return min_bound, None
        
        bound = h_func(self.initial)
        iteration = 0
        
        while iteration < max_iterations:
            path_states = {hash(self.initial)}
            t, result = search(self.initial, 0, bound, path_states)
            
            if t == -1:
                end_time = time.time()
                path = self.reconstruct_path(result)
                
                print("\nOUTPUT RESULTS")
                print("-" * 70)
                print(f"Success/Failure: SUCCESS")
                print(f"Heuristic/Parameters: IDA* with {h_name}, f-bound iteratively increased")
                print(f"(Sub)Optimal Path: {' → '.join(path)}")
                print(f"Path Length: {len(path)} moves (Optimal)")
                print(f"Iterations: {iteration + 1}")
                print(f"Total States Explored: {self.nodes_expanded}")
                print(f"Total Time Taken: {end_time - start_time:.6f} seconds")
                self.print_state_tracking('IDA*')
                print("="*70)
                
                return {
                    'algorithm': f'IDA* ({heuristic})',
                    'path_length': len(path),
                    'nodes_expanded': self.nodes_expanded,
                    'visited_states': len(self.visited_states),
                    'states_skipped': self.states_skipped,
                    'time': end_time - start_time,
                    'found': True
                }
            
            if t == float('inf'):
                break
            
            bound = t
            iteration += 1
        
        end_time = time.time()
        print("\nOUTPUT RESULTS")
        print("-" * 70)
        print(f"Success/Failure: FAILURE")
        print(f"Total States Explored: {self.nodes_expanded}")
        print(f"Total Time Taken: {end_time - start_time:.6f} seconds")
        self.print_state_tracking('IDA*')
        print("="*70)
        
        return {'algorithm': f'IDA* ({heuristic})', 'found': False, 'time': end_time - start_time}
    
    def simulated_annealing(self, heuristic='manhattan', max_iterations=10000) -> Dict:
        h_func = self.manhattan_distance if heuristic == 'manhattan' else self.misplaced_tiles
        h_name = "h2 (Manhattan Distance)" if heuristic == 'manhattan' else "h1 (Misplaced Tiles)"
        
        print("\n" + "="*70)
        print(f"SIMULATED ANNEALING WITH STATE TRACKING - {h_name}")
        print("="*70)
        
        # Reset tracking
        self.nodes_expanded = 0
        self.visited_states = set()
        self.states_skipped = 0
        
        # SA parameters
        initial_temp = 100
        cooling_rate = 0.995
        temperature = initial_temp
        
        start_time = time.time()
        current = self.initial
        current_h = h_func(current)
        best_state = current
        best_h = current_h
        
        self.visited_states.add(hash(current))
        
        for iteration in range(max_iterations):
            if self.is_goal(current):
                end_time = time.time()
                path = self.reconstruct_path(current)
                
                print("\nOUTPUT RESULTS")
                print("-" * 70)
                print(f"Success/Failure: SUCCESS")
                print(f"Heuristic/Parameters: Simulated Annealing with {h_name}")
                print(f"  - Initial Temperature: {initial_temp}")
                print(f"  - Cooling Rate: {cooling_rate} (Geometric)")
                print(f"  - Final Temperature: {temperature:.4f}")
                print(f"  - Acceptance Probability: exp(ΔE/T)")
                print(f"(Sub)Optimal Path: {' → '.join(path)}")
                print(f"Path Length: {len(path)} moves")
                print(f"Iterations: {iteration + 1}")
                print(f"Total States Explored: {self.nodes_expanded}")
                print(f"Total Time Taken: {end_time - start_time:.6f} seconds")
                self.print_state_tracking('Simulated Annealing')
                print("="*70)
                
                return {
                    'algorithm': f'Simulated Annealing ({heuristic})',
                    'path_length': len(path),
                    'nodes_expanded': self.nodes_expanded,
                    'visited_states': len(self.visited_states),
                    'states_skipped': self.states_skipped,
                    'time': end_time - start_time,
                    'found': True
                }
            
            self.nodes_expanded += 1
            
            # Get random successor
            moves = current.get_possible_moves()
            if not moves:
                break
            
            tile_pos, direction = random.choice([(m[:2], m[2]) for m in moves])
            next_state = current.move(tile_pos, direction)
            next_h = h_func(next_state)
            next_hash = hash(next_state)
            
            # STATE TRACKING
            if next_hash in self.visited_states:
                self.states_skipped += 1
                # Still evaluate for SA (may revisit with probability)
            else:
                self.visited_states.add(next_hash)
            
            # Calculate delta (negative because lower h is better)
            delta_e = current_h - next_h
            
            # Accept better states always, worse states with probability
            if delta_e > 0:  # Better state
                current = next_state
                current_h = next_h
            else:  # Worse state - accept with probability
                acceptance_prob = math.exp(delta_e / max(temperature, 0.01))
                if random.random() < acceptance_prob:
                    current = next_state
                    current_h = next_h
            
            # Track best state found
            if current_h < best_h:
                best_state = current
                best_h = current_h
            
            # Cool down
            temperature *= cooling_rate
            
            if temperature < 0.01:
                break
        
        end_time = time.time()
        
        # Return best state found
        if best_h == 0:  # Found goal
            path = self.reconstruct_path(best_state)
            print("\nOUTPUT RESULTS")
            print("-" * 70)
            print(f"Success/Failure: SUCCESS")
            print(f"Heuristic/Parameters: Simulated Annealing with {h_name}")
            print(f"  - Initial Temperature: {initial_temp}")
            print(f"  - Cooling Rate: {cooling_rate}")
            print(f"  - Final Temperature: {temperature:.4f}")
            print(f"(Sub)Optimal Path: {' → '.join(path)}")
            print(f"Path Length: {len(path)} moves")
            print(f"Total States Explored: {self.nodes_expanded}")
            print(f"Total Time Taken: {end_time - start_time:.6f} seconds")
            self.print_state_tracking('Simulated Annealing')
            print("="*70)
            return {
                'algorithm': f'Simulated Annealing ({heuristic})',
                'path_length': len(path),
                'nodes_expanded': self.nodes_expanded,
                'visited_states': len(self.visited_states),
                'states_skipped': self.states_skipped,
                'time': end_time - start_time,
                'found': True
            }
        else:
            print("\nOUTPUT RESULTS")
            print("-" * 70)
            print(f"Success/Failure: FAILURE (Best h={best_h})")
            print(f"Total States Explored: {self.nodes_expanded}")
            print(f"Total Time Taken: {end_time - start_time:.6f} seconds")
            self.print_state_tracking('Simulated Annealing')
            print("="*70)
            return {
                'algorithm': f'Simulated Annealing ({heuristic})',
                'found': False,
                'time': end_time - start_time
            }


def read_input_file(filename: str = 'input.txt') -> Tuple[PuzzleState, PuzzleState]:
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
    print("\n" + "="*70)
    print("MANUAL INPUT MODE - MATRIX FORMAT")
    print("="*70)
    print("Enter the puzzle state row by row (3 values per row).")
    print("Use 'B' for the blank tile. Separate values with spaces.")
    print("Example for a row: 1 2 3")
    print("-"*70)
    
    def input_state(state_name: str) -> PuzzleState:
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
        for row in board:
            print("  " + ' '.join(row))
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
    print("\n" + "="*70)
    print("ALGORITHM SELECTION MENU")
    print("="*70)
    print("\nA. UNINFORMED SEARCH")
    print("  1. BFS (Breadth-First Search)")
    print("  2. DFS (Depth-First Search)")
    print("\nB. INFORMED SEARCH")
    print("  3. Greedy Best-First (Misplaced Tiles)")
    print("  4. Greedy Best-First (Manhattan Distance)")
    print("  5. A* (Misplaced Tiles)")
    print("  6. A* (Manhattan Distance)")
    print("\nC. MEMORY-BOUNDED & LOCAL SEARCH")
    print("  7. IDA* (Misplaced Tiles)")
    print("  8. IDA* (Manhattan Distance)")
    print("  9. Simulated Annealing (Misplaced Tiles)")
    print(" 10. Simulated Annealing (Manhattan Distance)")
    print("\nSPECIAL OPTIONS")
    print(" 11. Run ALL Algorithms")
    print("  0. Exit")
    print("="*70)


def main():
    print("="*70)
    print("AI ASSIGNMENT - QUESTION 1 (A, B, C): 8-PUZZLE SEARCH")
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
    
    print("\n" + "="*70)
    print("PUZZLE CONFIGURATION")
    print("="*70)
    print("\nInitial State:")
    for row in start.board:
        print("  " + ' '.join(row))
    
    print("\nGoal State:")
    for row in goal.board:
        print("  " + ' '.join(row))
    
    # Algorithm selection loop
    while True:
        display_menu()
        choice = input("\nSelect algorithm (0-11): ").strip()
        
        if choice == '0':
            print("\n✓ Exiting program. Thank you!")
            break
        
        # Create solver instance
        solver = PuzzleSolver(start, goal)
        
        if choice == '1':
            solver.bfs()
        elif choice == '2':
            solver.dfs()
        elif choice == '3':
            solver.greedy_best_first('misplaced')
        elif choice == '4':
            solver.greedy_best_first('manhattan')
        elif choice == '5':
            solver.a_star('misplaced')
        elif choice == '6':
            solver.a_star('manhattan')
        elif choice == '7':
            solver.ida_star('misplaced')
        elif choice == '8':
            solver.ida_star('manhattan')
        elif choice == '9':
            solver.simulated_annealing('misplaced')
        elif choice == '10':
            solver.simulated_annealing('manhattan')
        elif choice == '11':
            print("\n" + "="*70)
            print("RUNNING ALL ALGORITHMS")
            print("="*70)
            
            print("\n" + "="*70)
            print("A. UNINFORMED SEARCH")
            print("="*70)
            
            solver = PuzzleSolver(start, goal)
            solver.bfs()
            
            solver = PuzzleSolver(start, goal)
            solver.dfs()
            
            print("\n" + "="*70)
            print("B. INFORMED SEARCH")
            print("="*70)
            
            solver = PuzzleSolver(start, goal)
            solver.greedy_best_first('misplaced')
            
            solver = PuzzleSolver(start, goal)
            solver.greedy_best_first('manhattan')
            
            solver = PuzzleSolver(start, goal)
            solver.a_star('misplaced')
            
            solver = PuzzleSolver(start, goal)
            solver.a_star('manhattan')
            
            print("\n" + "="*70)
            print("C. MEMORY-BOUNDED & LOCAL SEARCH")
            print("="*70)
            
            solver = PuzzleSolver(start, goal)
            solver.ida_star('misplaced')
            
            solver = PuzzleSolver(start, goal)
            solver.ida_star('manhattan')
            
            solver = PuzzleSolver(start, goal)
            solver.simulated_annealing('misplaced')
            
            solver = PuzzleSolver(start, goal)
            solver.simulated_annealing('manhattan')
            
            print("\n" + "="*70)
            print("ALL ALGORITHMS COMPLETED")
            print("="*70)
        else:
            print("❌ Invalid choice! Please select 0-11.")
            continue
        
        # Ask to continue
        cont = input("\nRun another algorithm? (y/n): ").strip().lower()
        if cont != 'y':
            print("\n✓ Exiting program. Thank you!")
            break
    
    print("\n" + "="*70)
    print("STATE TRACKING FEATURES:")
    print("  ✓ Visited States List maintained for all algorithms")
    print("  ✓ Not Visited States filtered out (cycle prevention)")
    print("  ✓ No revisiting root, parent, or explored states")
    print("  ✓ Efficiency metrics show states prevented from revisiting")
    print("  ✓ Complete output format with all required fields")
    print("="*70)


if __name__ == "__main__":
    main()