# Load Data
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import os
import pandas as pd
import concavity


def edge_hole_detection_from_arrays(node_ids, x, y, min_hole_size=0.5, max_hole_size=20.0, combine_output=False, output_dir='results'):
    """
    Detect edge and hole nodes from arrays of node IDs and coordinates.

    Parameters:
        node_ids (array): Array of node IDs.
        x (array): X-coordinates of nodes.
        y (array): Y-coordinates of nodes.
        min_hole_size (float): Minimum hole size for detection.
        max_hole_size (float): Maximum hole size for detection.
        combine_output (bool): Whether to combine edge and hole nodes in one CSV file.
        output_dir (str): Directory to save results.

    Returns:
        tuple: Arrays of edge node IDs and hole node IDs.
    """
    points = np.column_stack((x, y))

    # Get boundary nodes
    edge_node_ids = get_boundary_nodes_concave(points, node_ids, k=5)  # Adjust k as needed

    # Get hole nodes
    hole_node_ids = find_holes(points, node_ids, min_hole_size, max_hole_size)

    # Remove boundary nodes from hole nodes
    hole_node_ids = np.setdiff1d(hole_node_ids, edge_node_ids)

    # Save results
    save_results(output_dir, edge_node_ids, hole_node_ids, combine_output)

    # Visualize results
    visualize_results(x, y, node_ids, edge_node_ids, hole_node_ids, output_dir)

    print(f"Found {len(edge_node_ids)} edge nodes and {len(hole_node_ids)} hole nodes")

    return edge_node_ids, hole_node_ids


def get_boundary_nodes_concave(points, node_ids, k=5):
    """
    Get boundary nodes using a concave hull.

    Parameters:
        points (array): Array of node coordinates.
        node_ids (array): Array of node IDs.
        k (int): Parameter for concave hull generation.

    Returns:
        array: Array of boundary node IDs.
    """
    ch = concavity.concave_hull(points, k)
    boundary_coords = np.array(ch.exterior.coords)
    boundary_node_ids = []

    for bx, by in boundary_coords:
        distances = np.linalg.norm(points - np.array([bx, by]), axis=1)
        closest_idx = np.argmin(distances)
        boundary_node_ids.append(node_ids[closest_idx])

    return np.unique(boundary_node_ids)


def find_holes(points, node_ids, min_hole_size=0.5, max_hole_size=20.0):
    """
    Detect holes based on circumcircle radius of Delaunay triangles.

    Parameters:
        points (array): Array of node coordinates.
        node_ids (array): Array of node IDs.
        min_hole_size (float): Minimum hole size for detection.
        max_hole_size (float): Maximum hole size for detection.

    Returns:
        array: Array of hole node IDs.
    """
    tri = Delaunay(points)
    hole_candidates = []

    for simplex in tri.simplices:
        vertices = points[simplex]
        a, b, c = np.linalg.norm(vertices[1] - vertices[0]), np.linalg.norm(vertices[2] - vertices[1]), np.linalg.norm(vertices[0] - vertices[2])
        s = (a + b + c) / 2
        area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))

        if area > 0:
            circum_r = (a * b * c) / (4 * area)
            if min_hole_size <= circum_r <= max_hole_size:
                hole_candidates.extend(simplex)

    hole_indices = np.unique(hole_candidates)
    return node_ids[hole_indices.astype(int)]


def save_results(output_dir, edge_node_ids, hole_node_ids, combine_output):
    """
    Save edge and hole node results to CSV files.

    Parameters:
        output_dir (str): Directory to save results.
        edge_node_ids (array): Array of edge node IDs.
        hole_node_ids (array): Array of hole node IDs.
        combine_output (bool): Whether to combine edge and hole nodes in one CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)

    if combine_output:
        combined_ids = np.concatenate([edge_node_ids, hole_node_ids])
        pd.DataFrame({'node_id': combined_ids}).to_csv(os.path.join(output_dir, 'combined_nodes.csv'), index=False)
    else:
        pd.DataFrame({'node_id': edge_node_ids}).to_csv(os.path.join(output_dir, 'edge_nodes.csv'), index=False)
        pd.DataFrame({'node_id': hole_node_ids}).to_csv(os.path.join(output_dir, 'hole_nodes.csv'), index=False)


def visualize_results(x, y, node_ids, edge_node_ids, hole_node_ids, output_dir):
    """
    Visualize edge and hole nodes.

    Parameters:
        x (array): X-coordinates of nodes.
        y (array): Y-coordinates of nodes.
        node_ids (array): Array of node IDs.
        edge_node_ids (array): Array of edge node IDs.
        hole_node_ids (array): Array of hole node IDs.
        output_dir (str): Directory to save visualization.
    """
    plt.figure(figsize=(12, 10))
    plt.scatter(x, y, s=1, color='lightblue', label='All Nodes')

    edge_mask = np.isin(node_ids, edge_node_ids)
    plt.scatter(x[edge_mask], y[edge_mask], s=8, color='red', label='Edge Nodes')

    hole_mask = np.isin(node_ids, hole_node_ids)
    plt.scatter(x[hole_mask], y[hole_mask], s=8, color='green', label='Hole Nodes')

    plt.title('Edge and Hole Detection')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'node_detection.png'), dpi=300)
    plt.show()


def main():
    """
    Main function to load data and perform edge and hole detection.
    """
    try:
        experiment_number = input("Please enter the experiment number: ")
        experiment = int(experiment_number)
        base_path = f"results_preprocessing/experiment_{experiment}/"

        force_data = np.load(base_path + "force_data.npy")
        nodes_data = np.load(base_path + "nodes_data.npy", allow_pickle=True)
        time_data = np.load(base_path + "time_data.npy")

        print("First three rows of force_data:")
        print(force_data[:3])
        print("\nFirst three rows of nodes_data:")
        print(nodes_data[:3])
        print("\nFirst three rows of time_data:")
        print(time_data[:3])

        nodes_data_frame0 = nodes_data[0]  # Shape (3406, 5)
        node_ids = nodes_data_frame0[:, 0].astype(int)
        x = nodes_data_frame0[:, 1].astype(float)
        y = nodes_data_frame0[:, 2].astype(float)

        min_hole_size = float(input("Enter minimum hole size (default 0.5): ") or 0.5)
        max_hole_size = float(input("Enter maximum hole size (default 20.0): ") or 20.0)
        combine_output = input("Combine edge and hole nodes in one CSV file? (y/n, default n): ").lower() == 'y'

        edge_nodes, hole_nodes = edge_hole_detection_from_arrays(
            node_ids, x, y, min_hole_size, max_hole_size, combine_output, output_dir=base_path
        )
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
