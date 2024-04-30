import numpy as np

def generate_points_on_sphere(num_points, radius=1):
    # Step 1: Generate random points in a cube
    points = np.random.normal(size=(num_points, 3))
    
    # Step 2: Normalize the points to lie on the surface of a unit sphere
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
    
    # Step 3: Scale the points to the desired radius
    points *= radius
    
    return points

def generate_points_inside_sphere(num_points, radius=1):
    points = []
    while len(points) < num_points:
        # Generate a random point in a cube
        point = np.random.uniform(-radius, radius, size=3)
        
        # Check if the point is inside the sphere
        if np.linalg.norm(point) <= radius:
            points.append(point)
    
    return np.array(points)


def compute_distance_xyz(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return dist

num_points_base = 1000
radiuses = [1, 5, 10]
DISTS = []
for r, radius in enumerate(radiuses):
    num_points = num_points_base * radius
    # Example usage:
    points_contour = generate_points_on_sphere(num_points, radius=radius)
    points = generate_points_inside_sphere(num_points, radius=radius)
    xs = points[:,0]
    ys = points[:,1]
    zs = points[:,2]

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(xs, ys, zs)

    # xs = points_contour[:,0]
    # ys = points_contour[:,1]
    # zs = points_contour[:,2]
    # ax.scatter(xs, ys, zs)

    # plt.show()

    centroid = np.mean(points, axis=0)

    dists_contours = []
    dists_centroid = []
    for point in points:
        _dists = []
        for point_c in points_contour:
            dist = compute_distance_xyz(point, point_c)
            _dists.append(dist)

        dists_contours.append(np.min(_dists))
        dist = compute_distance_xyz(point, centroid)
        dists_centroid.append(dist)
        

    for point in points_contour:
        print(compute_distance_xyz(point, centroid))
    dists = np.array(dists_centroid) / (np.array(dists_centroid) + np.array(dists_contours))
    DISTS.append(dists)
    
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10,5))
for r, dists in enumerate(DISTS):
    radius = radiuses[r]
    n_hist, bins, patches = ax.hist(dists, alpha=0.5, bins=30, label="radius = {}".format(radius), density=True)
ax.legend()
ax.set_xlabel("relative position on sphere")
plt.show()

