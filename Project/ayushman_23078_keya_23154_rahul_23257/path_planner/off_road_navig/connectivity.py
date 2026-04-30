import pickle
from collections import deque
from tqdm import tqdm  # Run 'pip install tqdm' to use this

# 1. Load data
print("Loading graph data...")
with open('graph.pkl', 'rb') as f:
    data = pickle.load(f)
    nodes = data['nodes']
    edges = data['edges']

# 2. Build adjacency list with a progress bar
adj = {}
print("Building adjacency list...")
for edge in tqdm(edges, desc="Processing edges"):
    # Only use edges that are actually walkable/passable
    if hasattr(edge, 'is_passable') and not edge.is_passable:
        continue
        
    a, b = edge.src_id, edge.dst_id
    adj.setdefault(a, []).append(b)
    adj.setdefault(b, []).append(a)

# 3. Find connected components
visited = set()
components = []

def bfs(start):
    comp = set()
    # Using deque for O(1) pops, making this MUCH faster
    queue = deque([start])
    while queue:
        n = queue.popleft()
        if n in visited:
            continue
        visited.add(n)
        comp.add(n)
        for neighbor in adj.get(n, []):
            if neighbor not in visited:
                queue.append(neighbor)
    return comp

print("\nAnalyzing connectivity...")
# Progress bar based on number of nodes
with tqdm(total=len(nodes), desc="Grouping nodes") as pbar:
    for nid in nodes:
        if nid not in visited:
            comp_before = len(visited)
            components.append(bfs(nid))
            # Update progress bar by how many new nodes we just visited
            nodes_found = len(visited) - comp_before
            pbar.update(nodes_found)

# 4. Results
components.sort(key=len, reverse=True)
print(f"\nGraph analysis complete.")
print(f"Total Components found: {len(components)}")

for i, c in enumerate(components[:5]):
    print(f"  Component {i+1}: {len(c)} nodes")

# Check connectivity for specific IDs
start_id, goal_id = 8, 2689
start_comp = next((i for i, c in enumerate(components) if start_id in c), -1)
goal_comp = next((i for i, c in enumerate(components) if goal_id in c), -1)

print(f"\nStart (Node {start_id}) is in component: {start_comp}")
print(f"Goal (Node {goal_id}) is in component: {goal_comp}")

if start_comp == goal_comp and start_comp != -1:
    print("✅ PATH EXISTS: These nodes are connected.")
else:
    print("⚠️ DISCONNECTED: No path exists between these nodes.")