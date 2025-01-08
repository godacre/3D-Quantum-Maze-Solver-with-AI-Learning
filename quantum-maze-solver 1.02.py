import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Dict, Set

class QuantumMazeSolver:
    def __init__(self, size: int, obstacle_probability: float = 0.2, tunnel_probability: float = 0.05):
        """
        Initialize the Quantum Maze Solver
        
        Args:
            size (int): Dimensions of the 3D maze
            obstacle_probability (float): Probability of a cell being a wall
            tunnel_probability (float): Probability of a cell being a quantum tunnel
        """
        self.size = size
        self.maze = self._generate_maze(obstacle_probability, tunnel_probability)
        self.q_table = {}  # Q-learning state-action value table
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 1.0
        self.exploration_decay = 0.99
        
    def _generate_maze(self, obstacle_prob: float, tunnel_prob: float) -> np.ndarray:
        """
        Generate a 3D maze with start, end, quantum tunnels, and obstacles
        
        Returns:
            3D numpy array representing the maze
        """
        maze = np.full((self.size, self.size, self.size), fill_value='.')
        
        # Add random obstacles
        for x in range(self.size):
            for y in range(self.size):
                for z in range(self.size):
                    if random.random() < obstacle_prob:
                        maze[x, y, z] = '#'
        
        # Add quantum tunnels
        tunnel_count = 0
        while tunnel_count < 2:
            x, y, z = self._random_cell()
            if maze[x, y, z] == '.':
                maze[x, y, z] = 'Q'
                tunnel_count += 1
        
        # Add start and end points
        start_x, start_y, start_z = self._random_cell()
        end_x, end_y, end_z = self._random_cell()
        
        while (start_x == end_x and start_y == end_y and start_z == end_z) or \
              maze[start_x, start_y, start_z] != '.' or \
              maze[end_x, end_y, end_z] != '.':
            end_x, end_y, end_z = self._random_cell()
        
        maze[start_x, start_y, start_z] = 'S'
        maze[end_x, end_y, end_z] = 'E'
        
        return maze
    
    def _random_cell(self) -> Tuple[int, int, int]:
        """
        Generate a random 3D cell coordinate
        
        Returns:
            Tuple of (x, y, z) coordinates
        """
        return (
            random.randint(0, self.size - 1),
            random.randint(0, self.size - 1),
            random.randint(0, self.size - 1)
        )
    
    def _get_possible_moves(self, x: int, y: int, z: int) -> List[Tuple[int, int, int]]:
        """
        Get valid moves from current position considering maze boundaries and obstacles
        
        Args:
            x, y, z (int): Current position coordinates
        
        Returns:
            List of valid move coordinates
        """
        moves = [
            (x+dx, y+dy, z+dz) 
            for dx, dy, dz in [(0,0,1), (0,0,-1), (0,1,0), (0,-1,0), (1,0,0), (-1,0,0)]
        ]
        
        # Filter moves within maze bounds and not walls
        valid_moves = [
            (nx, ny, nz) for nx, ny, nz in moves 
            if (0 <= nx < self.size and 
                0 <= ny < self.size and 
                0 <= nz < self.size and 
                self.maze[nx, ny, nz] != '#')
        ]
        
        return valid_moves
    
    def _find_quantum_tunnels(self) -> List[Tuple[int, int, int]]:
        """
        Find all quantum tunnel locations
        
        Returns:
            List of quantum tunnel coordinates
        """
        return [
            (x, y, z) for x in range(self.size) 
            for y in range(self.size) 
            for z in range(self.size) 
            if self.maze[x, y, z] == 'Q'
        ]
    
    def q_learning_solve(self, max_iterations: int = 1000) -> List[Tuple[int, int, int]]:
        """
        Solve maze using Q-learning algorithm
        
        Args:
            max_iterations (int): Maximum learning iterations
        
        Returns:
            Optimal path through the maze
        """
        # Find start and end points
        start = tuple(np.argwhere(self.maze == 'S')[0])
        end = tuple(np.argwhere(self.maze == 'E')[0])
        
        for _ in range(max_iterations):
            current_state = start
            path = [current_state]
            
            while current_state != end:
                # Exploration vs exploitation
                if random.random() < self.exploration_rate:
                    # Random move
                    possible_moves = self._get_possible_moves(*current_state)
                    if not possible_moves:
                        break
                    next_state = random.choice(possible_moves)
                else:
                    # Best known move
                    possible_moves = self._get_possible_moves(*current_state)
                    if not possible_moves:
                        break
                    
                    # Get Q-values for possible moves
                    q_values = [
                        self.q_table.get((current_state, move), 0) 
                        for move in possible_moves
                    ]
                    next_state = possible_moves[np.argmax(q_values)]
                
                # Quantum tunnel handling
                if self.maze[next_state[0], next_state[1], next_state[2]] == 'Q':
                    tunnels = self._find_quantum_tunnels()
                    tunnels.remove(next_state)
                    next_state = random.choice(tunnels) if tunnels else next_state
                
                # Update Q-value
                reward = -1 if next_state != end else 100
                current_q = self.q_table.get((current_state, next_state), 0)
                max_future_q = max([
                    self.q_table.get((next_state, future_state), 0) 
                    for future_state in self._get_possible_moves(*next_state)
                ], default=0)
                
                new_q = (1 - self.learning_rate) * current_q + \
                        self.learning_rate * (reward + self.discount_factor * max_future_q)
                
                self.q_table[(current_state, next_state)] = new_q
                
                current_state = next_state
                path.append(current_state)
                
                if current_state == end:
                    break
            
            # Decay exploration rate
            self.exploration_rate *= self.exploration_decay
        
        return path
    
    def visualize_maze_3d(self, path: List[Tuple[int, int, int]] = None):
        """
        Create a 3D visualization of the maze
        
        Args:
            path (List[Tuple]): Optional path to highlight
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot maze cells
        for x in range(self.size):
            for y in range(self.size):
                for z in range(self.size):
                    color = 'white'
                    if self.maze[x, y, z] == '#':
                        color = 'black'
                    elif self.maze[x, y, z] == 'Q':
                        color = 'purple'
                    elif self.maze[x, y, z] == 'S':
                        color = 'green'
                    elif self.maze[x, y, z] == 'E':
                        color = 'red'
                    
                    ax.scatter(x, y, z, c=color, alpha=0.6, edgecolors='gray')
        
        # Plot path if provided
        if path:
            path_x, path_y, path_z = zip(*path)
            ax.plot(path_x, path_y, path_z, color='blue', linewidth=3, label='Optimal Path')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D Quantum Maze ({self.size}x{self.size}x{self.size})')
        plt.legend()
        plt.show()

def main():
    # Solve a 5x5x5 quantum maze
    maze_solver = QuantumMazeSolver(size=5, obstacle_probability=0.2, tunnel_probability=0.05)
    
    # Solve the maze using Q-learning
    optimal_path = maze_solver.q_learning_solve()
    
    # Visualize the maze and path
    maze_solver.visualize_maze_3d(optimal_path)
    
    # Print path details
    print(f"Path Length: {len(optimal_path)}")
    print(f"Path: {optimal_path}")

if __name__ == "__main__":
    main()
