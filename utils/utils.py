import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from collections import defaultdict, deque
import heapq
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
import tqdm

class OptimizedHouseRoutePlanner:
    def __init__(self, houses, streets, num_agents, start_point):
        """
        houses: dict, key = house_id, value = (x, y)
        streets: list of tuples (house1, house2, weight)
        num_agents: number of agents
        start_point: depot (house id) where each route starts/ends
        """
        self.houses = houses
        self.streets = streets
        self.num_agents = num_agents
        self.start_point = start_point
        self.graph = self._build_graph()
        self.best_paths = []  # list of routes (each route is a list of house ids)
        self.best_total_weight = float('inf')

    def _build_graph(self):
        graph = defaultdict(dict)
        for h1, h2, weight in self.streets:
            graph[h1][h2] = weight
            graph[h2][h1] = weight
        return graph

    def _bfs_check_connectivity(self):
        queue = deque([self.start_point])
        visited = {self.start_point}
        while queue:
            node = queue.popleft()
            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited == set(self.houses.keys())

    def visualize_grid(self, annotate=True):
        """
        Visualize the grid.

        - Draw streets as thicker, dark-gray lines with the travel time annotated at the midpoint.
        The width of each street is inversely proportional to its travel time weight: faster streets
        (with lower weight) appear wider.
        - Draw houses as translucent rectangles and with square markers.
        - For small grids (<=20 houses) and when annotate=True, annotate:
            * The depot with "Depot {id}".
            * Other houses with "House {id}\nT: {time}" where time is the shortest distance from the depot.
        """
        plt.figure(figsize=(10, 10))
        plt.style.use('seaborn-darkgrid')
        ax = plt.gca()

        # Determine min and max weights for scaling the street widths.
        all_weights = [w for (_, _, w) in self.streets]
        min_weight = min(all_weights)
        max_weight = max(all_weights)
        min_width = 1  # Narrowest street width.
        max_width = 5  # Widest street width.

        # Draw streets with width based on travel time.
        for h1, h2, weight in self.streets:
            x1, y1 = self.houses[h1]
            x2, y2 = self.houses[h2]
            # Avoid division by zero if all weights are equal.
            if max_weight != min_weight:
                lw = (max_weight - weight) / (max_weight - min_weight) * (max_width - min_width) + min_width
            else:
                lw = (min_width + max_width) / 2
            ax.plot([x1, x2], [y1, y2], color='dimgray', lw=lw, alpha=0.6, zorder=1)
            if annotate:
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                ax.text(mid_x, mid_y, f"{weight:.1f}", fontsize=8, color='blue',
                        ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'), zorder=2)

        # For small grids, compute distances from the depot for annotation.
        if annotate and len(self.houses) <= 20:
            dists = self.dijkstra(self.start_point)
        else:
            dists = {}

        # Draw houses.
        house_width = 2   # width of the rectangle
        house_height = 2  # height of the rectangle
        for house_id, (x, y) in self.houses.items():
            if house_id == self.start_point:
                color = 'green'
                label = f"Depot {house_id}"
            else:
                color = 'red'
                if annotate and dists:
                    label = f"House {house_id}\nT: {dists[house_id]:.1f}"
                else:
                    label = f"House {house_id}"
            rect = Rectangle((x - house_width/2, y - house_height/2), house_width, house_height,
                            edgecolor='black', facecolor=color, alpha=0.3, zorder=2)
            ax.add_patch(rect)
            ax.scatter(x, y, c=color, s=300, zorder=3, marker="s", edgecolors='black', linewidths=1.5)
            ax.text(x, y, label, fontsize=9, ha='center', va='center', color='white', zorder=4)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Optimized House Grid Visualization")
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.show()
    
    def dijkstra(self, source):
        """Compute shortest path distances from source to all nodes using Dijkstra's algorithm."""
        distances = {node: float('inf') for node in self.houses.keys()}
        distances[source] = 0
        visited = set()
        heap = [(0, source)]
        
        while heap:
            current_distance, current_node = heapq.heappop(heap)
            if current_node in visited:
                continue
            visited.add(current_node)
            for neighbor, weight in self.graph[current_node].items():
                new_distance = current_distance + weight
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    heapq.heappush(heap, (new_distance, neighbor))
        return distances

    def dijkstra_path(self, source, target):
        """
        Compute the shortest path between source and target using Dijkstra's algorithm.
        Returns the list of nodes (house ids) representing the path.
        """
        distances = {node: float('inf') for node in self.houses}
        distances[source] = 0
        prev = {node: None for node in self.houses}
        heap = [(0, source)]
        
        while heap:
            d, node = heapq.heappop(heap)
            if node == target:
                break
            if d > distances[node]:
                continue
            for neighbor, weight in self.graph[node].items():
                alt = d + weight
                if alt < distances[neighbor]:
                    distances[neighbor] = alt
                    prev[neighbor] = node
                    heapq.heappush(heap, (alt, neighbor))
        # Reconstruct the path.
        path = []
        cur = target
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

    def animate_agents(self):
        """
        Animate agents moving along their computed routes.
        The agents now move along the actual street paths (with linear interpolation along each edge)
        and leave a trail behind. Additionally, when an agent stands at a house (i.e., when it has
        arrived), it dwells there for a fixed duration and its marker color changes (to orange) to indicate
        that it is "dropping" something off.
        
        Modification: if a house has already been visited (i.e. an agent has already dwelled there),
        then a subsequent agent arriving at that same house will not incur the dwell time and can move
        on immediately.
        
        A time counter is displayed based on a constant speed factor.
        """
        if not self.best_paths:
            print("No routes to animate!")
            return

        # Setup figure.
        fig, ax = plt.subplots(figsize=(10, 10))
        x_vals = [coord[0] for coord in self.houses.values()]
        y_vals = [coord[1] for coord in self.houses.values()]
        ax.set_xlim(min(x_vals) - 5, max(x_vals) + 5)
        ax.set_ylim(min(y_vals) - 5, max(y_vals) + 5)
        ax.set_title("Agent Route Animation")

        # Determine street widths (as in visualize_grid).
        all_weights = [w for (_, _, w) in self.streets]
        min_weight = min(all_weights)
        max_weight = max(all_weights)
        min_width = 1
        max_width = 5

        # Draw streets with variable width.
        for h1, h2, weight in self.streets:
            x1, y1 = self.houses[h1]
            x2, y2 = self.houses[h2]
            if max_weight != min_weight:
                lw = (max_weight - weight) / (max_weight - min_weight) * (max_width - min_width) + min_width
            else:
                lw = (min_width + max_width) / 2
            ax.plot([x1, x2], [y1, y2], color='dimgray', lw=lw, alpha=0.6)
        
        # Draw houses as background rectangles.
        house_width = 1
        house_height = 1
        for house_id, (x, y) in self.houses.items():
            color = 'green' if house_id == self.start_point else 'red'
            rect = Rectangle((x - house_width/2, y - house_height/2), house_width, house_height,
                            edgecolor='black', facecolor=color, alpha=0.3, zorder=1)
            ax.add_patch(rect)

        # Setup interpolation parameters.
        frames_per_weight = 5
        dwell_frames = frames_per_weight  # Extra frames to dwell at each house.

        # Global set to record houses that have been visited (and hence already dwelled at)
        visited_houses = set()

        # Build the full interpolated paths for each agent.
        # Each position is stored as (x, y, is_dwell)
        agent_full_paths = []
        for route in self.best_paths:
            positions = []
            # Start at the depot.
            start_pos = self.houses[route[0]]
            positions.append((start_pos[0], start_pos[1], False))
            for i in range(len(route) - 1):
                source = route[i]
                target = route[i + 1]
                actual_path = self.dijkstra_path(source, target)
                for j in range(len(actual_path) - 1):
                    p_start = self.houses[actual_path[j]]
                    p_end = self.houses[actual_path[j + 1]]
                    edge_time = self.graph[actual_path[j]][actual_path[j+1]]
                    n_frames = max(1, int(edge_time * frames_per_weight))
                    for k in range(n_frames):
                        t = k / n_frames
                        x = p_start[0] + t * (p_end[0] - p_start[0])
                        y = p_start[1] + t * (p_end[1] - p_start[1])
                        positions.append((x, y, False))
                # Instead of always dwelling at the target house, check if it was visited before.
                house_coord = self.houses[target]
                if target not in visited_houses:
                    # First time visit: add full dwell frames.
                    for d in range(dwell_frames):
                        positions.append((house_coord[0], house_coord[1], True))
                    visited_houses.add(target)
                else:
                    # Already visited: add only one frame so that the agent does not consume extra time.
                    positions.append((house_coord[0], house_coord[1], False))
            agent_full_paths.append(positions)

        max_frames = max(len(p) for p in agent_full_paths)

        # Create scatter markers and trail lines for each agent.
        agent_plots = []
        trails = []
        colors = plt.cm.get_cmap('tab10', len(agent_full_paths))
        for i in range(len(agent_full_paths)):
            scatter = ax.scatter([], [], s=100, label=f'Agent {i}', zorder=5,
                                color=colors(i), edgecolors='black', linewidths=1.5)
            line, = ax.plot([], [], lw=2, linestyle='-', color=colors(i), alpha=0.7, zorder=4)
            agent_plots.append(scatter)
            trails.append(line)

        # Time counter.
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='black')

        def update(frame):
            for i, positions in enumerate(agent_full_paths):
                idx = min(frame, len(positions) - 1)
                x, y, is_dwell = positions[idx]
                agent_plots[i].set_offsets([[x, y]])
                # Change marker color during dwell to indicate a "drop" action.
                if is_dwell:
                    agent_plots[i].set_facecolor("orange")
                else:
                    agent_plots[i].set_facecolor(colors(i))
                # Update the trail.
                trail_x = [pos[0] for pos in positions[:idx + 1]]
                trail_y = [pos[1] for pos in positions[:idx + 1]]
                trails[i].set_data(trail_x, trail_y)
            # Update time display.
            current_time = frame / frames_per_weight
            time_text.set_text(f"Time: {current_time:.1f}")
            return agent_plots + trails + [time_text]

        anim = FuncAnimation(fig, update, frames=max_frames, interval=200, blit=True)

        anim.save("agent_route_animation.gif", writer="pillow", fps=5)
        
        ax.legend()
        plt.show()

    # --------------------------
    # Helper functions for clustering and TSP solving
    # --------------------------
    
    def compute_distance_matrix(self, nodes):
        """
        For a given list of nodes, compute a dictionary with keys (u, v)
        giving the shortest distance from u to v.
        """
        dist_matrix = {}
        for node in nodes:
            distances = self.dijkstra(node)
            for other in nodes:
                dist_matrix[(node, other)] = distances[other]
        return dist_matrix

    def solve_tsp_nearest_neighbor(self, nodes, dist_matrix):
        """
        Solve TSP for a set of nodes (including the depot) using a nearest neighbor heuristic.
        """
        current = self.start_point
        route = [current]
        remaining = set(nodes)
        remaining.discard(current)
        
        while remaining:
            next_node = min(remaining, key=lambda n: dist_matrix[(current, n)])
            route.append(next_node)
            remaining.remove(next_node)
            current = next_node
        # Optionally, return to the depot to complete the cycle.
        route.append(self.start_point)
        return route

    def two_opt(self, route, dist_matrix):
        """
        Improve an existing TSP route using the 2-opt algorithm.
        """
        improved = True
        best_route = route
        best_distance = self.route_distance(best_route, dist_matrix)
        
        while improved:
            improved = False
            for i in range(1, len(best_route) - 2):
                for j in range(i+1, len(best_route) - 1):
                    if j - i == 1:
                        continue
                    new_route = best_route[:i] + best_route[i:j][::-1] + best_route[j:]
                    new_distance = self.route_distance(new_route, dist_matrix)
                    if new_distance < best_distance:
                        best_route = new_route
                        best_distance = new_distance
                        improved = True
            # Exit when no improvement is found.
        return best_route

    def route_distance(self, route, dist_matrix):
        """Compute the total distance of a given route using the precomputed distance matrix."""
        total = 0
        for i in range(len(route)-1):
            total += dist_matrix[(route[i], route[i+1])]
        return total

    # --------------------------
    # Improved training via multiple restarts with enhanced clustering
    # --------------------------
    
    def train_improved(self, num_restarts=10, use_duration_clustering=True):
        """
        Runs several random restarts of the clustering + TSP process and picks the best overall solution.
        
        If use_duration_clustering is True, we incorporate the travel times (durations) into clustering:
          - Compute a pairwise distance matrix (using Dijkstra) among the houses.
          - Use MDS to embed houses into 2D space.
          - Run KMeans on the resulting embedding.
        Otherwise, KMeans is applied directly on the (x,y) coordinates.
        """
        if not self._bfs_check_connectivity():
            print("Graph is not fully connected!")
            return [], float('inf')
        
        best_routes = None
        best_weight = float('inf')
        
        for restart in tqdm.tqdm(range(num_restarts), desc="Training restarts"):
            # Get houses excluding the depot.
            house_ids = [hid for hid in self.houses.keys() if hid != self.start_point]
            if not house_ids:
                return [[self.start_point]], 0
            
            n_clusters = min(self.num_agents, len(house_ids))
            
            if use_duration_clustering:
                # Build a pairwise travel time distance matrix for the houses.
                n = len(house_ids)
                duration_matrix = np.zeros((n, n))
                for i, hid in enumerate(house_ids):
                    dists = self.dijkstra(hid)
                    for j, hid2 in enumerate(house_ids):
                        duration_matrix[i, j] = dists[hid2]
                # Use MDS to embed houses in 2D so that the travel times are preserved.
                mds = MDS(n_components=2, dissimilarity="precomputed", random_state=restart, normalized_stress='auto')
                embedded_coords = mds.fit_transform(duration_matrix)
                kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=restart)
                labels = kmeans.fit_predict(embedded_coords)
            else:
                # Use the raw (x,y) coordinates.
                coords = np.array([self.houses[hid] for hid in house_ids])
                kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=restart)
                labels = kmeans.fit_predict(coords)
            
            clusters = defaultdict(list)
            for hid, label in zip(house_ids, labels):
                clusters[label].append(hid)
            
            agent_routes = []
            total_weight = 0
            for label in clusters:
                # For each cluster, include the depot.
                nodes = clusters[label] + [self.start_point]
                nodes = list(set(nodes))  # Remove duplicates if any.
                dist_matrix = self.compute_distance_matrix(nodes)
                route = self.solve_tsp_nearest_neighbor(nodes, dist_matrix)
                improved_route = self.two_opt(route, dist_matrix)
                agent_routes.append(improved_route)
                total_weight += self.route_distance(improved_route, dist_matrix)
            
            if total_weight < best_weight:
                best_weight = total_weight
                best_routes = agent_routes
        
        self.best_paths = best_routes
        self.best_total_weight = best_weight
        return best_routes, best_weight

# --------------------------
# Main functions
# --------------------------

def main_complex_grid():
    """
    Create a complex grid with up to 2000 houses and (for example) 20 agents.
    """
    print("Running complex grid with up to 2000 houses...")
    # Generate 2000 houses with coordinates between 0 and 100.
    houses = {i: (random.randint(0, 100), random.randint(0, 100)) for i in range(2000)}
    
    # Create a chain to ensure connectivity.
    streets = []
    house_ids = list(houses.keys())
    random.shuffle(house_ids)
    for i in range(len(house_ids) - 1):
        streets.append((house_ids[i], house_ids[i+1], random.randint(1, 10)))
    # Add extra random streets.
    extra_streets = [(random.randint(0, 1999), random.randint(0, 1999), random.randint(1, 10)) for _ in range(5000)]
    streets.extend(extra_streets)
    
    num_agents = 20
    start_point = 0
    planner = OptimizedHouseRoutePlanner(houses, streets, num_agents, start_point)
    best_routes, best_weight = planner.train_improved(num_restarts=10, use_duration_clustering=True)
    
    print("\nBest routes for complex grid (showing first 10 nodes per route):")
    for idx, route in enumerate(best_routes):
        print(f"Agent {idx} route: {route[:10]} ... Total nodes in route: {len(route)}")
    print(f"Total weight: {best_weight}")
    
    # For complex grid, disable detailed annotation to avoid clutter.
    planner.visualize_grid(annotate=False)
    # Uncomment the line below to see the enhanced animation.
    # planner.animate_agents()

def main_small_grid():
    """
    Create a small grid with up to 10 houses and 2 agents.
    """
    print("Running small grid with up to 10 houses and 2 agents...")
    # Generate 10 houses with coordinates between 0 and 10.
    houses = {i: (random.randint(0, 10), random.randint(0, 10)) for i in range(10)}
    
    # Create a chain to ensure connectivity.
    streets = []
    house_ids = list(houses.keys())
    random.shuffle(house_ids)
    for i in range(len(house_ids) - 1):
        streets.append((house_ids[i], house_ids[i+1], random.randint(1, 10)))
    # Add extra random streets.
    extra_streets = [(random.randint(0, 9), random.randint(0, 9), random.randint(1, 10)) for _ in range(20)]
    streets.extend(extra_streets)
    
    num_agents = 2
    start_point = 0
    planner = OptimizedHouseRoutePlanner(houses, streets, num_agents, start_point)
    best_routes, best_weight = planner.train_improved(num_restarts=10, use_duration_clustering=True)
    
    print("\nBest routes for small grid:")
    for idx, route in enumerate(best_routes):
        print(f"Agent {idx} route: {route}")
    print(f"Total weight: {best_weight}")
    
    planner.visualize_grid(annotate=True)
    planner.animate_agents()

def main_five_houses():
    """
    Create a very small grid with 5 houses and 2 agents.
    The visualization here shows annotations for both houses and streets.
    """
    print("Running five-house grid with 2 agents...")
    # Generate 5 houses with coordinates between 0 and 10.
    houses = {i: (random.randint(0, 10), random.randint(0, 10)) for i in range(5)}
    
    # Create a chain to ensure connectivity.
    streets = []
    house_ids = list(houses.keys())
    random.shuffle(house_ids)
    for i in range(len(house_ids) - 1):
        streets.append((house_ids[i], house_ids[i+1], random.randint(1, 10)))
    # Add extra random streets.
    extra_streets = [(random.randint(0, 4), random.randint(0, 4), random.randint(1, 10)) for _ in range(10)]
    streets.extend(extra_streets)
    
    num_agents = 2
    start_point = 0
    planner = OptimizedHouseRoutePlanner(houses, streets, num_agents, start_point)
    best_routes, best_weight = planner.train_improved(num_restarts=10, use_duration_clustering=True)
    
    print("\nBest routes for five-house grid:")
    for idx, route in enumerate(best_routes):
        print(f"Agent {idx} route: {route}")
    print(f"Total weight: {best_weight}")
    
    planner.visualize_grid(annotate=True)
    planner.animate_agents()
