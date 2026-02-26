# Create a sample input.txt file
with open('input.txt', 'w') as f:
    f.write('Start: 1;2;3;4;B;5;6;7;8\n')
    f.write('Goal: 1;2;3;4;5;6;7;B;8\n')

print("input.txt created successfully with sample data.")

import heapq
import time
from collections import deque
from typing import List, Tuple, Set, Optional

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
    
    def __str__(self):
        result = ""
        for row in self.board:
            result += " ".join(str(x) for x in row) + "\n"
        return result

class PuzzleSolver:
    def __init__(self, initial_state: PuzzleState, goal_state: PuzzleState):
        self.initial = initial_state
        self.goal = goal_state
        self.nodes_expanded = 0
        self.max_fringe_size = 0
    
    def is_goal(self, state: PuzzleState) -> bool:
        #current state = goal state
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
    
    def bfs(self):
        print("\n=== Running BFS ===")
        start_time = time.time()
        if self.is_goal(self.initial):
            end_time = time.time()
            print(f"\n--- RESULTS ---")
            print(f"Success/Failure: SUCCESS")
            print(f"Heuristic/Parameters: Uninformed Search (BFS - FIFO Queue)")
            print(f"(Sub)Optimal Path: [No moves needed]")
            print(f"Path Length: 0 moves")
            print(f"Total States Explored: 0")
            print(f"Total Time Taken: {end_time - start_time:.4f} seconds")
            return {
                'algorithm': 'BFS',
                'path_length': 0,
                'nodes_expanded': 0,
                'max_fringe': 1,
                'time': end_time - start_time,
                'found': True
            }
        
        frontier = deque([self.initial])
        explored = set()
        in_frontier = {hash(self.initial)}  
        self.nodes_expanded = 0
        
        while frontier:
            self.max_fringe_size = max(self.max_fringe_size, len(frontier))
            state = frontier.popleft()
            
            if self.is_goal(state):
                end_time = time.time()
                path = self.reconstruct_path(state)
                print(f"\n--- RESULTS ---")
                print(f"Success/Failure: SUCCESS")
                print(f"Heuristic/Parameters: Uninformed Search (BFS - FIFO Queue)")
                print(f"(Sub)Optimal Path: {' -> '.join(path)}")
                print(f"Path Length: {len(path)} moves")
                print(f"Total States Explored: {self.nodes_expanded}")
                print(f"Max Fringe Size: {self.max_fringe_size}")
                print(f"Total Time Taken: {end_time - start_time:.4f} seconds")
                return {
                    'algorithm': 'BFS',
                    'path_length': len(path),
                    'nodes_expanded': self.nodes_expanded,
                    'max_fringe': self.max_fringe_size,
                    'time': end_time - start_time,
                    'found': True
                }
            
            explored.add(hash(state))
            self.nodes_expanded += 1
            
            for tile_row, tile_col, direction in state.get_possible_moves():
                child = state.move((tile_row, tile_col), direction)
                child_hash = hash(child)
                if child_hash not in explored and child_hash not in in_frontier:
                    frontier.append(child)
                    in_frontier.add(child_hash)
        
        end_time = time.time()
        print(f"\n--- RESULTS ---")
        print(f"Success/Failure: FAILURE")
        print(f"Heuristic/Parameters: Uninformed Search (BFS - FIFO Queue)")
        print(f"(Sub)Optimal Path: No solution found")
        print(f"Total States Explored: {self.nodes_expanded}")
        print(f"Total Time Taken: {end_time - start_time:.4f} seconds")
        return {
            'algorithm': 'BFS',
            'path_length': 0,
            'nodes_expanded': self.nodes_expanded,
            'max_fringe': self.max_fringe_size,
            'time': end_time - start_time,
            'found': False
        }
    
    def a_star(self, heuristic='manhattan'):
        print(f"\n=== Running A* with {heuristic} heuristic ===")
        start_time = time.time()
        
        h_func = self.manhattan_distance if heuristic == 'manhattan' else self.misplaced_tiles
        
        frontier = []
        heapq.heappush(frontier, (h_func(self.initial), 0, self.initial))
        explored = set()
        self.nodes_expanded = 0
        counter = 1
        
        while frontier:
            self.max_fringe_size = max(self.max_fringe_size, len(frontier))
            _, _, state = heapq.heappop(frontier)
            
            if self.is_goal(state):
                end_time = time.time()
                path = self.reconstruct_path(state)
                h_name = "h1 (Misplaced Tiles)" if heuristic == 'misplaced' else "h2 (Manhattan Distance)"
                print(f"\n--- RESULTS ---")
                print(f"Success/Failure: SUCCESS")
                print(f"Heuristic/Parameters: A* Search with {h_name}, f(n) = g(n) + h(n)")
                print(f"(Sub)Optimal Path: {' -> '.join(path)}")
                print(f"Path Length: {len(path)} moves (Optimal)")
                print(f"Total States Explored: {self.nodes_expanded}")
                print(f"Max Fringe Size: {self.max_fringe_size}")
                print(f"Total Time Taken: {end_time - start_time:.4f} seconds")
                return {
                    'algorithm': f'A* ({heuristic})',
                    'path_length': len(path),
                    'nodes_expanded': self.nodes_expanded,
                    'max_fringe': self.max_fringe_size,
                    'time': end_time - start_time,
                    'found': True
                }
            
            explored.add(hash(state))
            self.nodes_expanded += 1
            
            for tile_row, tile_col, direction in state.get_possible_moves():
                child = state.move((tile_row, tile_col), direction)
                if hash(child) not in explored:
                    f_score = child.cost + h_func(child)
                    heapq.heappush(frontier, (f_score, counter, child))
                    counter += 1
        
        end_time = time.time()
        h_name = "h1 (Misplaced Tiles)" if heuristic == 'misplaced' else "h2 (Manhattan Distance)"
        print(f"\n--- RESULTS ---")
        print(f"Success/Failure: FAILURE")
        print(f"Heuristic/Parameters: A* Search with {h_name}, f(n) = g(n) + h(n)")
        print(f"(Sub)Optimal Path: No solution found")
        print(f"Total States Explored: {self.nodes_expanded}")
        print(f"Total Time Taken: {end_time - start_time:.4f} seconds")
        return {
            'algorithm': f'A* ({heuristic})',
            'path_length': 0,
            'nodes_expanded': self.nodes_expanded,
            'max_fringe': self.max_fringe_size,
            'time': end_time - start_time,
            'found': False
        }
    
    def dfs(self, depth_limit=50):
        print(f"\n=== Running DFS (depth limit: {depth_limit}) ===")
        start_time = time.time()
        
        frontier = [self.initial]
        explored = set()
        self.nodes_expanded = 0
        
        while frontier:
            self.max_fringe_size = max(self.max_fringe_size, len(frontier))
            state = frontier.pop()  # LIFO for DFS
            
            if self.is_goal(state):
                end_time = time.time()
                path = self.reconstruct_path(state)
                print(f"\n--- RESULTS ---")
                print(f"Success/Failure: SUCCESS")
                print(f"Heuristic/Parameters: Uninformed Search (DFS - LIFO Stack, Depth Limit={depth_limit})")
                print(f"(Sub)Optimal Path: {' -> '.join(path)}")
                print(f"Path Length: {len(path)} moves (May be suboptimal)")
                print(f"Total States Explored: {self.nodes_expanded}")
                print(f"Max Fringe Size: {self.max_fringe_size}")
                print(f"Total Time Taken: {end_time - start_time:.4f} seconds")
                return {
                    'algorithm': 'DFS',
                    'path_length': len(path),
                    'nodes_expanded': self.nodes_expanded,
                    'max_fringe': self.max_fringe_size,
                    'time': end_time - start_time,
                    'found': True
                }
            
            if state.cost >= depth_limit:
                continue
            
            explored.add(hash(state))
            self.nodes_expanded += 1
            
            for tile_row, tile_col, direction in state.get_possible_moves():
                child = state.move((tile_row, tile_col), direction)
                if hash(child) not in explored:
                    frontier.append(child)
        
        end_time = time.time()
        print(f"\n--- RESULTS ---")
        print(f"Success/Failure: FAILURE")
        print(f"Heuristic/Parameters: Uninformed Search (DFS - LIFO Stack, Depth Limit={depth_limit})")
        print(f"(Sub)Optimal Path: No solution found within depth limit")
        print(f"Total States Explored: {self.nodes_expanded}")
        print(f"Total Time Taken: {end_time - start_time:.4f} seconds")
        return {
            'algorithm': 'DFS',
            'path_length': 0,
            'nodes_expanded': self.nodes_expanded,
            'max_fringe': self.max_fringe_size,
            'time': end_time - start_time,
            'found': False
        }
    
    def greedy_best_first(self, heuristic='manhattan'):
        print(f"\n=== Running Greedy Best-First with {heuristic} heuristic ===")
        start_time = time.time()
        
        h_func = self.manhattan_distance if heuristic == 'manhattan' else self.misplaced_tiles
        
        frontier = []
        heapq.heappush(frontier, (h_func(self.initial), 0, self.initial))
        explored = set()
        self.nodes_expanded = 0
        counter = 1
        
        while frontier:
            self.max_fringe_size = max(self.max_fringe_size, len(frontier))
            _, _, state = heapq.heappop(frontier)
            
            if self.is_goal(state):
                end_time = time.time()
                path = self.reconstruct_path(state)
                h_name = "h1 (Misplaced Tiles)" if heuristic == 'misplaced' else "h2 (Manhattan Distance)"
                print(f"\n--- RESULTS ---")
                print(f"Success/Failure: SUCCESS")
                print(f"Heuristic/Parameters: Greedy Best-First Search with {h_name}")
                print(f"(Sub)Optimal Path: {' -> '.join(path)}")
                print(f"Path Length: {len(path)} moves (May be suboptimal)")
                print(f"Total States Explored: {self.nodes_expanded}")
                print(f"Max Fringe Size: {self.max_fringe_size}")
                print(f"Total Time Taken: {end_time - start_time:.4f} seconds")
                return {
                    'algorithm': f'Greedy ({heuristic})',
                    'path_length': len(path),
                    'nodes_expanded': self.nodes_expanded,
                    'max_fringe': self.max_fringe_size,
                    'time': end_time - start_time,
                    'found': True
                }
            
            explored.add(hash(state))
            self.nodes_expanded += 1
            
            for tile_row, tile_col, direction in state.get_possible_moves():
                child = state.move((tile_row, tile_col), direction)
                if hash(child) not in explored:
                    h_score = h_func(child)
                    heapq.heappush(frontier, (h_score, counter, child))
                    counter += 1
        
        end_time = time.time()
        h_name = "h1 (Misplaced Tiles)" if heuristic == 'misplaced' else "h2 (Manhattan Distance)"
        print(f"\n--- RESULTS ---")
        print(f"Success/Failure: FAILURE")
        print(f"Heuristic/Parameters: Greedy Best-First Search with {h_name}")
        print(f"(Sub)Optimal Path: No solution found")
        print(f"Total States Explored: {self.nodes_expanded}")
        print(f"Total Time Taken: {end_time - start_time:.4f} seconds")
        return {
            'algorithm': f'Greedy ({heuristic})',
            'path_length': 0,
            'nodes_expanded': self.nodes_expanded,
            'max_fringe': self.max_fringe_size,
            'time': end_time - start_time,
            'found': False
        }
    
    def ida_star(self, heuristic='manhattan', max_iterations=100):
        print(f"\n=== Running IDA* with {heuristic} heuristic ===")
        print(f"[Memory-Bounded Search: Depth-bound adjustments per iteration]")
        start_time = time.time()
        
        h_func = self.manhattan_distance if heuristic == 'manhattan' else self.misplaced_tiles
        
        self.nodes_expanded = 0
        threshold = h_func(self.initial)
        print(f"Initial threshold (f-cost bound): {threshold}")
        
        for iteration in range(max_iterations):
            result, new_threshold = self._ida_star_search(self.initial, 0, threshold, h_func, set())
            
            if result is not None:
                end_time = time.time()
                path = self.reconstruct_path(result)
                h_name = "h1 (Misplaced Tiles)" if heuristic == 'misplaced' else "h2 (Manhattan Distance)"
                print(f"\n--- RESULTS ---")
                print(f"Success/Failure: SUCCESS")
                print(f"Heuristic/Parameters: IDA* with {h_name}, Iterations={iteration + 1}, Final Threshold={threshold}")
                if len(path) > 0:
                    print(f"(Sub)Optimal Path: {' -> '.join(path)}")
                else:
                    print(f"(Sub)Optimal Path: [No moves needed]")
                print(f"Path Length: {len(path)} moves (Optimal)")
                print(f"Total States Explored: {self.nodes_expanded}")
                print(f"Total Time Taken: {end_time - start_time:.4f} seconds")
                return {
                    'algorithm': f'IDA* ({heuristic})',
                    'path_length': len(path),
                    'nodes_expanded': self.nodes_expanded,
                    'max_fringe': 'N/A',
                    'time': end_time - start_time,
                    'found': True
                }
            
            if new_threshold == float('inf'):
                print(f"\nIteration {iteration + 1}: No solution within threshold {threshold}")
                break
            
            print(f"Iteration {iteration + 1}: Threshold = {threshold}, Next threshold = {new_threshold}")
            threshold = new_threshold
        
        end_time = time.time()
        h_name = "h1 (Misplaced Tiles)" if heuristic == 'misplaced' else "h2 (Manhattan Distance)"
        print(f"\n--- RESULTS ---")
        print(f"Success/Failure: FAILURE")
        print(f"Heuristic/Parameters: IDA* with {h_name}, Max Iterations={max_iterations}")
        print(f"(Sub)Optimal Path: No solution found within iteration limit")
        print(f"Total States Explored: {self.nodes_expanded}")
        print(f"Total Time Taken: {end_time - start_time:.4f} seconds")
        return {
            'algorithm': f'IDA* ({heuristic})',
            'path_length': 0,
            'nodes_expanded': self.nodes_expanded,
            'max_fringe': 'N/A',
            'time': end_time - start_time,
            'found': False
        }
    
    def _ida_star_search(self, state, g, threshold, h_func, path):
        self.nodes_expanded += 1
        f = g + h_func(state)
        
        if f > threshold:
            return None, f
        
        if self.is_goal(state):
            return state, threshold
        
        min_threshold = float('inf')
        path.add(hash(state))
        
        for tile_row, tile_col, direction in state.get_possible_moves():
            child = state.move((tile_row, tile_col), direction)
            
            if hash(child) not in path:
                result, new_threshold = self._ida_star_search(child, g + 1, threshold, h_func, path)
                
                if result is not None:
                    return result, new_threshold
                
                min_threshold = min(min_threshold, new_threshold)
        
        path.remove(hash(state))
        return None, min_threshold
    
    def simulated_annealing(self, heuristic='manhattan', initial_temp=100, cooling_rate=0.95, max_iterations=10000):
        print(f"\n=== Running Simulated Annealing with {heuristic} heuristic ===")
        print(f"[Local Search: Escaping local maxima via probabilistic acceptance]")
        print(f"Cooling Schedule: T(0) = {initial_temp}, α = {cooling_rate}, T(n+1) = α × T(n)")
        print(f"Acceptance Probability: P(accept worse) = exp(-ΔE/T)")
        start_time = time.time()
        
        import random
        import math
        
        h_func = self.manhattan_distance if heuristic == 'manhattan' else self.misplaced_tiles
        
        current = self.initial
        temperature = initial_temp
        self.nodes_expanded = 0
        worse_moves_accepted = 0  
        better_moves = 0
        
        for iteration in range(max_iterations):
            self.nodes_expanded += 1
            
            if self.is_goal(current):
                end_time = time.time()
                path = self.reconstruct_path(current)
                h_name = "h1 (Misplaced Tiles)" if heuristic == 'misplaced' else "h2 (Manhattan Distance)"
                print(f"\n--- RESULTS ---")
                print(f"Success/Failure: SUCCESS")
                print(f"Heuristic/Parameters: Simulated Annealing with {h_name}, T0={initial_temp}, α={cooling_rate}, Iterations={iteration + 1}")
                if len(path) > 0:
                    print(f"(Sub)Optimal Path: {' -> '.join(path)}")
                else:
                    print(f"(Sub)Optimal Path: [No moves needed]")
                print(f"Path Length: {len(path)} moves (May be suboptimal)")
                print(f"Total States Explored: {self.nodes_expanded}")
                print(f"Better Moves: {better_moves}, Worse Moves Accepted: {worse_moves_accepted}")
                print(f"Final Temperature: {temperature:.4f}")
                print(f"Total Time Taken: {end_time - start_time:.4f} seconds")
                return {
                    'algorithm': f'Sim Anneal ({heuristic})',
                    'path_length': len(path),
                    'nodes_expanded': self.nodes_expanded,
                    'max_fringe': 'N/A',
                    'time': end_time - start_time,
                    'found': True
                }
            
            moves = current.get_possible_moves()
            if not moves:
                break
            
            tile_row, tile_col, direction = random.choice(moves)
            neighbor = current.move((tile_row, tile_col), direction)
            
            current_h = h_func(current)
            neighbor_h = h_func(neighbor)
            delta_e = neighbor_h - current_h
            
            if delta_e < 0:
                current = neighbor
                better_moves += 1
            elif temperature > 0:
                acceptance_probability = math.exp(-delta_e / temperature)
                if random.random() < acceptance_probability:
                    current = neighbor
                    worse_moves_accepted += 1  
          
            temperature *= cooling_rate
            
            if temperature < 0.01:
                break
        
        end_time = time.time()
        h_name = "h1 (Misplaced Tiles)" if heuristic == 'misplaced' else "h2 (Manhattan Distance)"
        print(f"\n--- RESULTS ---")
        print(f"Success/Failure: FAILURE")
        print(f"Heuristic/Parameters: Simulated Annealing with {h_name}, T0={initial_temp}, α={cooling_rate}")
        print(f"(Sub)Optimal Path: No solution found (Best h={h_func(current)})")
        print(f"Total States Explored: {self.nodes_expanded}")
        print(f"Better Moves: {better_moves}, Worse Moves Accepted: {worse_moves_accepted}")
        print(f"Final Temperature: {temperature:.4f}")
        print(f"Total Time Taken: {end_time - start_time:.4f} seconds")
        return {
            'algorithm': f'Sim Anneal ({heuristic})',
            'path_length': 0,
            'nodes_expanded': self.nodes_expanded,
            'max_fringe': 'N/A',
            'time': end_time - start_time,
            'found': False
        }

def parse_input(input_str: str) -> List[List]:
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

def read_from_file(filename: str = 'input.txt') -> Tuple[List[List], List[List]]:
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        start_line = None
        goal_line = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('Start:'):
                start_line = line.split('Start:')[1].strip()
            elif line.startswith('Goal:'):
                goal_line = line.split('Goal:')[1].strip()
        
        if start_line and goal_line:
            start_board = parse_file_format(start_line)
            goal_board = parse_file_format(goal_line)
            return start_board, goal_board
        else:
            return None, None
    except FileNotFoundError:
        return None, None

def display_results_table(results: List[dict]):
    if not results:
        return
    
    print("\n" + "=" * 100)
    print("ALGORITHM PERFORMANCE COMPARISON TABLE")
    print("=" * 100)
  
    print(f"{'Algorithm':<25} {'Solution':<10} {'Path Len':<10} {'Nodes Exp':<12} {'Max Fringe':<12} {'Time (s)':<10}")
    print("-" * 100)
    
    sorted_results = sorted(results, key=lambda x: x['time'])
    
    for result in sorted_results:
        solution = "✓ Yes" if result['found'] else "✗ No"
        path_len = str(result['path_length']) if result['found'] else "-"
        max_fringe = str(result['max_fringe']) if result['max_fringe'] != 'N/A' else 'N/A'
        
        print(f"{result['algorithm']:<25} {solution:<10} {path_len:<10} {result['nodes_expanded']:<12} {max_fringe:<12} {result['time']:<10.4f}")
    
    print("=" * 100)
    
    successful = [r for r in results if r['found']]
    if successful:
        print("\nSUMMARY:")
        print(f"  Algorithms that found solution: {len(successful)}/{len(results)}")
        avg_time = sum(r['time'] for r in successful) / len(successful)
        avg_nodes = sum(r['nodes_expanded'] for r in successful) / len(successful)
        print(f"  Average time (successful): {avg_time:.4f} seconds")
        print(f"  Average nodes expanded (successful): {avg_nodes:.0f}")
        
        fastest = min(successful, key=lambda x: x['time'])
        most_efficient = min(successful, key=lambda x: x['nodes_expanded'])
        print(f"  Fastest algorithm: {fastest['algorithm']} ({fastest['time']:.4f}s)")
        print(f"  Most efficient (fewest nodes): {most_efficient['algorithm']} ({most_efficient['nodes_expanded']} nodes)")
    print("=" * 100)

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

def main():
    print("=" * 70)
    print("8-PUZZLE Search Algorithms Implementation --AI")
    print("=" * 70)
   
    print("\nSELECT INPUT MODE:")
    print("1. Manual Input (Enter puzzle row by row)")
    print("2. Read from input.txt file")
    print("=" * 70)
    
    mode = input("\nEnter your choice (1 or 2): ").strip()
    
    if mode == '2':
        initial_board, goal_board = read_from_file('input.txt')
        
        if initial_board is None:
            print("\n❌ Could not read from input.txt file!")
            print("Switching to manual input mode...\n")
            mode = '1'
        else:
            print("\n✓ Successfully read from input.txt")
            print("\n✓ Initial state:")
            for row in initial_board:
                print(" ".join(str(x) for x in row))
            print("\n✓ Goal state:")
            for row in goal_board:
                print(" ".join(str(x) for x in row))
    
    if mode == '1':
        print("\n" + "=" * 70)
        print("MANUAL INPUT MODE")
        print("=" * 70)
        print("\nEnter the INITIAL puzzle configuration:")
        print("- Enter 3 rows, each with 3 elements")
        print("- Use numbers 1-8 and 'B' for blank")
        print("- Separate elements with spaces")
        print("\nExample:")
        print("1 2 3")
        print("4 B 5")
        print("6 7 8")
        print("\nEnter your initial state:")
        
        initial_input = []
        for i in range(3):
            row = input(f"Row {i+1}: ").strip()
            initial_input.append(row)
        
        initial_board = parse_input('\n'.join(initial_input))
        
        if not validate_board(initial_board):
            print("\n❌ Invalid initial board! Please ensure:")
            print("- 3x3 grid")
            print("- Exactly one 'B'")
            print("- Numbers 1-8 (each appearing once)")
            return
        
        print("\n✓ Initial state:")
        for row in initial_board:
            print(" ".join(str(x) for x in row))
        
        print("\nEnter the GOAL puzzle configuration:")
        print("(Same format as initial state)")
        
        goal_input = []
        for i in range(3):
            row = input(f"Row {i+1}: ").strip()
            goal_input.append(row)
        
        goal_board = parse_input('\n'.join(goal_input))
        
        if not validate_board(goal_board):
            print("\n❌ Invalid goal board!")
            return
        
        print("\n✓ Goal state:")
        for row in goal_board:
            print(" ".join(str(x) for x in row))
    
    initial_state = PuzzleState(initial_board)
    goal_state = PuzzleState(goal_board)
    solver = PuzzleSolver(initial_state, goal_state)
   
    results = []
   
    print("\n" + "=" * 70)
    print("SELECT SEARCH ALGORITHM:")
    print("=" * 70)
    print("A. UNINFORMED SEARCH:")
    print("  1. BFS (Breadth-First Search)")
    print("  2. DFS (Depth-First Search)")
    print("\nB. INFORMED SEARCH:")
    print("  3. Greedy Best-First (h1 - Misplaced Tiles)")
    print("  4. Greedy Best-First (h2 - Manhattan Distance)")
    print("  5. A* (h1 - Misplaced Tiles)")
    print("  6. A* (h2 - Manhattan Distance)")
    print("\nC. MEMORY-BOUNDED & LOCAL SEARCH:")
    print("  7. IDA* (h1 - Misplaced Tiles)")
    print("  8. IDA* (h2 - Manhattan Distance)")
    print("  9. Simulated Annealing (h1 - Misplaced Tiles)")
    print("  10. Simulated Annealing (h2 - Manhattan Distance)")
    print("\nALL ALGORITHMS:")
    print("  11. Run all algorithms")
    print("=" * 70)
    
    choice = input("\nEnter your choice (1-11): ").strip()
    
    print("\n" + "=" * 70)
    
    if choice == '1':
        result = solver.bfs()
        if result:
            results.append(result)
    elif choice == '2':
        result = solver.dfs()
        if result:
            results.append(result)
    elif choice == '3':
        result = solver.greedy_best_first('misplaced')
        if result:
            results.append(result)
    elif choice == '4':
        result = solver.greedy_best_first('manhattan')
        if result:
            results.append(result)
    elif choice == '5':
        result = solver.a_star('misplaced')
        if result:
            results.append(result)
    elif choice == '6':
        result = solver.a_star('manhattan')
        if result:
            results.append(result)
    elif choice == '7':
        result = solver.ida_star('misplaced')
        if result:
            results.append(result)
    elif choice == '8':
        result = solver.ida_star('manhattan')
        if result:
            results.append(result)
    elif choice == '9':
        result = solver.simulated_annealing('misplaced')
        if result:
            results.append(result)
    elif choice == '10':
        result = solver.simulated_annealing('manhattan')
        if result:
            results.append(result)
    elif choice == '11':
        print("\nRUNNING ALL REQUIRED ALGORITHMS...")
        
        print("\n" + "=" * 70)
        print("A. UNINFORMED SEARCH")
        print("=" * 70)
        result = solver.bfs()
        if result:
            results.append(result)
        print("\n" + "-" * 70)
        solver = PuzzleSolver(initial_state, goal_state)
        result = solver.dfs()
        if result:
            results.append(result)
        
        print("\n" + "=" * 70)
        print("B. INFORMED SEARCH")
        print("=" * 70)
        solver = PuzzleSolver(initial_state, goal_state)
        result = solver.greedy_best_first('misplaced')
        if result:
            results.append(result)
        print("\n" + "-" * 70)
        solver = PuzzleSolver(initial_state, goal_state)
        result = solver.greedy_best_first('manhattan')
        if result:
            results.append(result)
        print("\n" + "-" * 70)
        solver = PuzzleSolver(initial_state, goal_state)
        result = solver.a_star('misplaced')
        if result:
            results.append(result)
        print("\n" + "-" * 70)
        solver = PuzzleSolver(initial_state, goal_state)
        result = solver.a_star('manhattan')
        if result:
            results.append(result)
        
        print("\n" + "=" * 70)
        print("C. MEMORY-BOUNDED & LOCAL SEARCH")
        print("=" * 70)
        solver = PuzzleSolver(initial_state, goal_state)
        result = solver.ida_star('misplaced')
        if result:
            results.append(result)
        print("\n" + "-" * 70)
        solver = PuzzleSolver(initial_state, goal_state)
        result = solver.ida_star('manhattan')
        if result:
            results.append(result)
        print("\n" + "-" * 70)
        solver = PuzzleSolver(initial_state, goal_state)
        result = solver.simulated_annealing('misplaced')
        if result:
            results.append(result)
        print("\n" + "-" * 70)
        solver = PuzzleSolver(initial_state, goal_state)
        result = solver.simulated_annealing('manhattan')
        if result:
            results.append(result)
    else:
        print("Invalid choice!")
   
    if results:
        display_results_table(results)
    
    print("\n" + "=" * 18)
    print("EXECUTION COMPLETE")
    print("=" * 18)

if __name__ == "__main__":
    main()
