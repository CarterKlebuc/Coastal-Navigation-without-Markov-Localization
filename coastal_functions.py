import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import heapq
import math

def calculate_map_entropy(map, world_size):
    information_map = np.zeros((world_size[0], world_size[1]))
    # Initial Information Map Generation without accounting for non-wall obstacles
    for i in range(world_size[0]):
        for j in range(world_size[1]):
            if map[i, j] == 1:
                information_map[i, j] = 1  # Walls have high information content
            else:
                # Distance to nearest wall
                distance_to_wall = calculate_closest_wall_distance(map, [i, j])
                information_map[i, j] = 1 / (distance_to_wall + 1)  # Inverse of distance to wall

    return information_map

def calculate_closest_wall_distance(matrix, start):
    rows = len(matrix)
    cols = len(matrix[0])

    # Define the directions for moving in the grid (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # Queue for BFS: stores the position and the current distance
    queue = deque([(start[0], start[1], 0)])

    # Set to track visited cells
    visited = set()
    visited.add((start[0], start[1]))

    # BFS to find the closest '1'
    while queue:
        x, y, dist = queue.popleft()

        # If we found a '1', return the distance
        if matrix[x][y] == 1:
            return dist

        # Explore neighbors
        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Check if the new position is within bounds and not visited
            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny, dist + 1))

    # If no '1' is found, return -1 (assuming there's always a '1' in the matrix)
    return -1

class Node:
    def __init__(self, x, y, cost=0, heuristic=0, parent=None):
        self.x = x
        self.y = y
        self.cost = cost  # g(n): cost from start to the current node
        self.heuristic = heuristic  # h(n): estimated cost from current node to goal
        self.parent = parent

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))


def a_star(start, goal, grid, information_map):
    def heuristic(a, b):
        # Using Euclidean distance as the heuristic function
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    # Define possible movements (up, down, left, right, and diagonals)
    movements = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]

    open_set = []
    heapq.heappush(open_set, Node(start[0], start[1], 0, heuristic(Node(start[0], start[1]), Node(goal[0], goal[1]))))

    came_from = {}
    cost_so_far = {start: 0}

    while open_set:
        current = heapq.heappop(open_set)

        if (current.x, current.y) == goal:
            path = []
            while current:
                path.append((current.x, current.y))
                current = current.parent
            return path[::-1]  # Return reversed path from start to goal

        for movement in movements:
            neighbor_x, neighbor_y = current.x + movement[0], current.y + movement[1]

            # Check if the neighbor is within bounds and is not an obstacle
            if 0 <= neighbor_x < len(grid) and 0 <= neighbor_y < len(grid[0]) and grid[neighbor_x][neighbor_y] == 0:
                #print("Info Map Current")
                #print(str(information_map[current.x, current.y]))
                cost_multiplier = 1000
                new_cost = cost_so_far[(current.x, current.y)] + 1 + (cost_multiplier * (1 - information_map[current.x, current.y])) # Assume all movements have the same cost

                if (neighbor_x, neighbor_y) not in cost_so_far or new_cost < cost_so_far[(neighbor_x, neighbor_y)]:
                    cost_so_far[(neighbor_x, neighbor_y)] = new_cost
                    priority = new_cost + heuristic(Node(neighbor_x, neighbor_y), Node(goal[0], goal[1]))
                    heapq.heappush(open_set, Node(neighbor_x, neighbor_y, new_cost, priority, current))
                    came_from[(neighbor_x, neighbor_y)] = (current.x, current.y)

    return None  # No path found

def create_experiment(start, goal):
    # Example occupancy grid: 0 represents free space, 1 represents obstacles
    world_size = [20, 20]
    occupancy_grid = np.zeros([world_size[0], world_size[1]])
    occupancy_grid[:, 0] = 1
    occupancy_grid[:, world_size[1] - 1] = 1
    occupancy_grid[0, :] = 1
    occupancy_grid[world_size[0] - 1, :] = 1
    occupancy_grid[6, 7] = 1
    occupancy_grid[7, 6] = 1
    occupancy_grid[7, 7] = 1
    occupancy_grid[6, 8] = 1
    occupancy_grid[7, 8] = 1
    occupancy_grid[5, 1] = 1
    occupancy_grid[4, 4] = 1

    occupancy_grid[10, 10] = 1
    occupancy_grid[11, 10] = 1
    occupancy_grid[10, 11] = 1
    occupancy_grid[11, 11] = 1

    occupancy_grid[13, 13] = 1
    occupancy_grid[14, 13] = 1
    occupancy_grid[13, 14] = 1
    occupancy_grid[14, 14] = 1



    information_map = calculate_map_entropy(occupancy_grid, world_size)

    new_goal = (goal[1], goal[0])
    new_start = (start[1], start[0])
    path = a_star(new_start, new_goal, occupancy_grid, information_map)

    if path:
        #print("Path found:")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=6)
        # Create a grid to represent the occupancy_grid
        grid_array = np.array(information_map)

        # Color map for visualization
        info_plot = ax1.imshow(grid_array)
        plt.colorbar(info_plot, ax = ax1, shrink = 0.45)


        ax1.set_title("Information Map")

        # Create a grid to represent the occupancy_grid
        grid_array = np.array(occupancy_grid)

        ax2.imshow(grid_array, cmap='gray_r')
        ax2.set_title("Coastal Navigation Path")

        # Plot the path
        if path:
            path_x = [x for x, y in path]
            path_y = [y for x, y in path]
            ax2.plot(path_y, path_x, color='blue', linewidth=2, label='Path')

        # Mark start and goal points
        ax2.scatter(start[0], start[1], color='green', marker='o', label='Start')
        ax2.scatter(goal[0], goal[1], color='red', marker='x', label='Goal')
        #plt.colorbar()

        # Legend and grid lines
        ax2.legend()
        plt.show()
    else:
        #print("No path found.")
        pass
