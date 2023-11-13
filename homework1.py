import numpy as np
import matplotlib.pyplot as plt

def distance_point_centroid(point, centroid):
    return np.sqrt(np.sum((point - centroid) ** 2))

def compute_centroids(points, labels, k):
    centroids = np.zeros((k, 2))
    for i in range(k):
        cluster_points = points[labels == i]
        centroids[i] = np.mean(cluster_points, axis=0)
    return centroids

def assign_labels(points, centroids):
    labels = np.zeros(len(points))
    for i, point in enumerate(points):
        distances = [distance_point_centroid(point, centroid) for centroid in centroids]
        labels[i] = np.argmin(distances)
    return labels

def compute_inertia(points, labels, centroids):
    inertia = 0
    for i, point in enumerate(points):
        inertia += distance_point_centroid(point, centroids[int(labels[i])])**2
    return inertia

def kmeans(points, k, steps, plot=False):
    centroids = np.random.rand(k, 2)

    if plot:
        plt.figure(figsize=(8, 8))

    for step in range(steps):
        labels = assign_labels(points, centroids)

        if plot:
            plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='rainbow')
            plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x')
            plt.title(f'Step {step+1}')
            plt.show()

        new_centroids = compute_centroids(points, labels, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

num_points = 200
points = np.random.rand(num_points, 2)

max_clusters = 10
list = []
for k in range(1, max_clusters+1):
    centroids, labels = kmeans(points, k, steps=10)
    inertia = compute_inertia(points, labels, centroids)
    list.append(inertia)

plt.plot(range(1, max_clusters+1), list, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow method to determine optimal number of clusters')
plt.show()

k_optimal = np.argmin(np.diff(list, 2)) + 2
print(f'Optimal number of clusters: {k_optimal}')

centroids, labels = kmeans(points, k_optimal, steps=10, plot=True)