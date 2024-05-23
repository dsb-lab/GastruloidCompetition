import numpy as np

from numba import njit, prange

@njit
def compute_distance_xyz(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return dist

@njit
def compute_distances(points, points_contour, centroid):
    
    len_points = len(points)
    len_points_contour = len(points_contour)

    dists_contours = np.zeros(len_points)
    dists_centroid = np.zeros(len_points)
    
    for p in prange(len_points):
        point = points[p]
        _dists = np.zeros(len_points_contour)
        for pc in prange(len_points_contour):
            point_c = points_contour[pc]
            dist = compute_distance_xyz(point, point_c)
            _dists[pc] = dist

        dists_contours[p] = np.min(_dists)
        dist = compute_distance_xyz(point, centroid)
        dists_centroid[p] = dist

    return dists_contours, dists_centroid

def generate_points_on_ellipsoid(semi_x=5, semi_y=3, semi_z=2, num_points=100):
    import numpy as np

    # Parameters for the ellipsoid
    a = semi_x  # Semi-major axis length along x-axis
    b = semi_y  # Semi-major axis length along y-axis
    c = semi_z  # Semi-major axis length along z-axis

    # Create a mesh grid for the ellipsoid
    u = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(0, np.pi, num_points)
    u, v = np.meshgrid(u, v)

    # Parametric equations for the ellipsoid
    x = a * np.cos(u) * np.sin(v)
    y = b * np.sin(u) * np.sin(v)
    z = c * np.cos(v)

    # Flatten the arrays to create a list of points
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()

    # Combine the points into a single array
    points = np.vstack((x_flat, y_flat, z_flat)).T

    return points


def generate_points_inside_ellipsoid(semi_x=5, semi_y=3, semi_z=2, num_points=10000):
    
    # Parameters for the ellipsoid
    a = semi_x  # Semi-major axis length along x-axis
    b = semi_y  # Semi-major axis length along y-axis
    c = semi_z  # Semi-major axis length along z-axis

    # Generate random points inside a unit sphere using spherical coordinates
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    cos_theta = np.random.uniform(-1, 1, num_points)
    u = np.random.uniform(0, 1, num_points)

    theta = np.arccos(cos_theta)
    r = u ** (1/3)

    # Convert spherical coordinates to Cartesian coordinates
    x_sphere = r * np.sin(theta) * np.cos(phi)
    y_sphere = r * np.sin(theta) * np.sin(phi)
    z_sphere = r * np.cos(theta)

    # Scale points to fit inside the ellipsoid
    x_ellipsoid = a * x_sphere
    y_ellipsoid = b * y_sphere
    z_ellipsoid = c * z_sphere

    # Combine the points into a single array
    points = np.vstack((x_ellipsoid, y_ellipsoid, z_ellipsoid)).T

    return points


num_points_base = 2000
radiuses = [[1,1,1], [2,1,1], [4,1,1]]
DISTS = []
for r, radius in enumerate(radiuses):

    print(radius)
    semi_x, semi_y, semi_z = radius
    num_points = np.rint(num_points_base * np.mean(radius)).astype("int64")

    print(num_points)
    # Example usage:
    points_contour = generate_points_on_ellipsoid(semi_x, semi_y, semi_z, num_points)
    points = generate_points_inside_ellipsoid(semi_x, semi_y, semi_z, num_points)
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

    dists_contours, dists_centroid = compute_distances(points, points_contour, centroid)

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

