import numpy as np
import math

def distance(a, b):
    return np.linalg.norm(a - b)

def merge_maps(map1, map2):
    merged = map1.copy()
    rows, cols = map1.shape

    for i in range(rows):
        for j in range(cols):
            v1 = map1[i, j]
            v2 = map2[i, j]

            if v1 == 0.5 and v2 != 0.5:
                merged[i, j] = v2
            elif v1 != 0.5 and v2 == 0.5:
                merged[i, j] = v1
            elif v1 != 0.5 and v2 != 0.5:
                merged[i, j] = max(v1, v2)

    return merged


def communicate(robots, edges, r_comm=10):
    for i in range(len(robots)):
        for j in range(i+1, len(robots)):

            r1 = robots[i]
            r2 = robots[j]

            dist = np.linalg.norm(r1.pos - r2.pos)
            edge = (r1.id, r2.id)

            if dist < r_comm and edge not in edges:
                edges.add(edge)

                merged = merge_maps(r1.map, r2.map)
                r1.map = merged.copy()
                r2.map = merged.copy()

                print(f"EDGE ADDED {edge} → maps merged")

            elif dist >= r_comm and edge in edges:
                edges.remove(edge)
                print(f"EDGE REMOVED {edge}")


def compute_Hr(point, robots, ds):
    Nr = len(robots)
    Hr = 0

    for r in robots:
        d = np.linalg.norm(r.pos - np.array(point))

        if d < ds:
            Hr += math.log(Nr) * (1 / (d + 1e-6))

    return Hr

ds = 5

circle_offsets = [
    (i, j)
    for i in range(-ds, ds+1)
    for j in range(-ds, ds+1)
    if i*i + j*j <= ds*ds
]

def compute_Hf(centroids, robot_pos, Nr):
    Hf_values = []

    if len(centroids) == 0:
        return Hf_values

    NC = len(centroids)

    for c in centroids:
        q = np.array(c["center"])
        Cq = c["size"]

        d = np.linalg.norm(q - robot_pos) + 1e-6

        Hf = - (Cq / d) * math.log(NC * Cq)

        Hf_values.append((q, Hf))

    return Hf_values

def sense(robot, global_map):
    x, y = int(robot.pos[0]), int(robot.pos[1])
    rows, cols = global_map.shape

    for dx, dy in circle_offsets:
        nx, ny = x + dx, y + dy

        if 0 <= nx < rows and 0 <= ny < cols:
            robot.map[nx, ny] = float(global_map[nx, ny])

            if global_map[nx, ny] == 1:
                continue


def find_frontiers(robot_map):
    frontiers = []
    rows, cols = robot_map.shape

    for i in range(1, rows-1):
        for j in range(1, cols-1):

            if np.isclose(robot_map[i, j], 0):
                neighbors = robot_map[i-1:i+2, j-1:j+2]

                if np.any(np.isclose(neighbors, 0.5)):
                    frontiers.append((i, j))

    filtered = []
    FRONTIER_MIN_DIST = 2

    for f in frontiers:
        if all(np.linalg.norm(np.array(f) - np.array(g)) > FRONTIER_MIN_DIST for g in filtered):
            filtered.append(f)

    return filtered

def cluster_frontiers(frontiers):
    clusters = []
    visited = set()

    for f in frontiers:
        if tuple(f) in visited:
            continue

        cluster = [f]
        queue = [f]
        visited.add(tuple(f))

        while queue:
            x, y = queue.pop(0)

            for fx, fy in frontiers:
                if (fx, fy) not in visited:
                    if abs(fx - x) <= 2 and abs(fy - y) <= 2:
                        visited.add((fx, fy))
                        queue.append((fx, fy))
                        cluster.append((fx, fy))

        clusters.append(cluster)

    centroids = []
    for cluster in clusters:
        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        cx = int(np.mean(xs))
        cy = int(np.mean(ys))

        centroids.append({
            "center": (cx, cy),
            "size": len(cluster),
            "points": cluster
        })

    return centroids

def is_reachable(robot, goal):
    x0, y0 = robot.pos
    x1, y1 = goal

    steps = int(np.linalg.norm([x1 - x0, y1 - y0]))
    if steps == 0:
        return True

    rows, cols = robot.map.shape

    for t in range(steps):
        alpha = t / steps
        x = int(x0 + alpha * (x1 - x0))
        y = int(y0 + alpha * (y1 - y0))

        # bounds check (important)
        if not (0 <= x < rows and 0 <= y < cols):
            return False

        # obstacle blocks path
        if robot.map[x, y] >= 0.9:
            return False

    return True
def compute_Htotal(centroids, robot, robots, ds):
    Nr = len(robots)

    Hf_vals = compute_Hf(centroids, robot.pos, Nr)

    if not Hf_vals:
        return None

    best = None
    best_val = float('inf')

    for q, Hf in Hf_vals:
        Hr = compute_Hr(q, robots, ds)

        Htotal = Hf + Hr

        # reachability penalty
        if not is_reachable(robot, q):
            Htotal += 500

        if Htotal < best_val:
            best_val = Htotal
            best = q

    return best

def assign_goal(robot, robots, ds=5):
    # STEP 1: find frontiers
    frontiers = find_frontiers(robot.map)
    print(f"robot {robot.id} frontiers found: {len(frontiers)}")

    if len(frontiers) == 0:
        return None

    # STEP 2: cluster → centroids
    centroids = cluster_frontiers(frontiers)

    # STEP 3: compute best candidate
    g_new = compute_Htotal(centroids, robot, robots, ds)

    if g_new is None:
        return None

    g_new = np.array(g_new, dtype=int)

    # ============================
    # STUCK CONDITION
    # ============================
    if robot.stuck_counter > 5:
        robot.g_cur = g_new
        robot.prev_pos = robot.pos.copy()
        robot.steps_since_goal = 0
        robot.stuck_counter = 0

        print(f"robot {robot.id} was stuck → forcing new goal {robot.g_cur}")
        return robot.g_cur

    # -------------------------------
    # INITIAL ASSIGNMENT
    # -------------------------------
    if robot.g_cur is None:
        robot.g_cur = g_new
        robot.prev_pos = robot.pos.copy()
        robot.steps_since_goal = 0

        print(f"robot {robot.id} assigned NEW goal {robot.g_cur}")
        return robot.g_cur

    # -------------------------------
    # GOAL UPDATE CONDITIONS
    # -------------------------------
    d_to_goal = np.linalg.norm(robot.pos - robot.g_cur)

    # reached goal
    if d_to_goal < 1:
        robot.g_cur = g_new
        robot.prev_pos = robot.pos.copy()
        robot.steps_since_goal = 0

        print(f"robot {robot.id} reached goal → new goal {robot.g_cur}")
        return robot.g_cur

    # time-based reassignment
    robot.steps_since_goal += 1

    d = np.linalg.norm(robot.prev_pos - robot.g_cur)
    kref = 0.1
    tref = max(1, int(kref * d / robot.vmax))

    if robot.steps_since_goal >= tref:
        robot.g_cur = g_new
        robot.prev_pos = robot.pos.copy()
        robot.steps_since_goal = 0

        print(f"robot {robot.id} reassigning goal → {robot.g_cur}")

    return robot.g_cur
def move(robot, robots, global_map):   # 🔥 pass global_map

    if robot.g_cur is None:
        return
    
    N = robot.map.shape[0]

    direction = robot.g_cur - robot.pos
    norm = np.linalg.norm(direction)
    if norm == 0:
        robot.steps_since_goal = 0
        return

    neighbors = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            cx = int(max(0, min(N-1, robot.pos[0] + dx)))
            cy = int(max(0, min(N-1, robot.pos[1] + dy)))
            neighbors.append(np.array([cx, cy], dtype=int))

    uniq = []
    for n in neighbors:
        if not any((n == u).all() for u in uniq):
            uniq.append(n)

    def adj_has_obstacle(px, py, use_map):
        x0 = max(0, px-1); x1 = min(N-1, px+1)
        y0 = max(0, py-1); y1 = min(N-1, py+1)
        block = use_map[x0:x1+1, y0:y1+1]
        return np.any(block >= 0.9)

    ROBOT_MIN_DIST = 1.5

    scored = []
    for p in uniq:
        px, py = int(p[0]), int(p[1])
        
        r_val = float(robot.map[px, py])
        
        if r_val >= 0.9:
            continue

        score = np.linalg.norm(robot.g_cur - p)

        if r_val == 0.0:
            score -= 5.0
        elif r_val == 0.5:
            score += 1.0

        if adj_has_obstacle(px, py, robot.map):
            score += 10.0
        elif adj_has_obstacle(px, py, global_map):
            score += 5.0

        for r in robots:
            if r.id != robot.id:
                dist = np.linalg.norm(r.pos - p)
                if dist < ROBOT_MIN_DIST:
                    score += 100

        scored.append((score, p))

    if not scored:
        for p in uniq:
            px, py = int(p[0]), int(p[1])
            if global_map[px, py] < 0.9:
                robot.pos = np.array([px, py], dtype=int)
                return
        return

    scored.sort(key=lambda s: s[0])
    best = scored[0][1]

    robot.prev_pos = robot.pos.copy()
    robot.pos = np.array([int(best[0]), int(best[1])], dtype=int)

    # stuck detection
    if np.all(robot.pos == robot.last_pos):
        robot.stuck_counter += 1
    else:
        robot.stuck_counter = 0

    robot.last_pos = robot.pos.copy()
