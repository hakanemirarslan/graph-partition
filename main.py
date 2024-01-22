#karger min cut adım adım gösteriyor kl çalışıyor
import sys
import networkx as nx
import random
import matplotlib.pyplot as plt
import community
import copy


class Edge:
    def __init__(self, s, d):
        self.src = s
        self.dest = d

# a class to represent a connected, undirected
# and unweighted graph as a collection of edges.
class Graph:
    # V-> Number of vertices, E-> Number of edges
    def __init__(self, v, e):
        self.V = v
        self.E = e
        # graph is represented as an array of edges.
        # Since the graph is undirected, the edge
        # from src to dest is also edge from dest
        # to src. Both are counted as 1 edge here.
        self.edge = []

# A class to represent a subset for union-find
class subset:
    def __init__(self, p, r):
        self.parent = p
        self.rank = r

# A very basic implementation of Karger's randomized
# algorithm for finding the minimum cut.

# A utility function to find set of an element i
# (uses path compression technique)
def find(subsets, i):

    # find root and make root as parent of i
    # (path compression)
    if subsets[i].parent != i:
        subsets[i].parent = find(subsets, subsets[i].parent)

    return subsets[i].parent

def Union(subsets, x, y):
    xroot = find(subsets, x)
    yroot = find(subsets, y)

    # Attach the smaller rank tree under the root
    # of the higher rank tree (Union by Rank)
    if subsets[xroot].rank < subsets[yroot].rank:
        subsets[xroot].parent = yroot
    elif subsets[xroot].rank > subsets[yroot].rank:
        subsets[yroot].parent = xroot
    else:
        # If ranks are the same, then make one as root
        # and increment its rank by one
        subsets[yroot].parent = xroot
        subsets[xroot].rank += 1

# #çalışan kl
# def kernighan_lin(graph):
#     total_nodes = len(graph.nodes())
#     partition_a = set(random.sample(graph.nodes(), total_nodes // 2))
#     partition_b = set(graph.nodes()) - partition_a

#     # Calculate initial cut size
#     initial_cut_size = nx.cut_size(graph, partition_a, partition_b)

#     while True:
#         improvement = False

#         # Iterate over vertices and try swapping between partitions
#         for node in partition_a:
#             gain = sum(1 for neighbor in graph.neighbors(node) if neighbor in partition_b) - \
#                    sum(1 for neighbor in graph.neighbors(node) if neighbor in partition_a)

#             # Find a node in partition_b to swap with
#             swap_candidate = max(partition_b, key=lambda x: len(set(graph.neighbors(x)) & partition_a))

#             if gain > 0:
#                 partition_a.remove(node)
#                 partition_b.remove(swap_candidate)
#                 partition_a.add(swap_candidate)
#                 partition_b.add(node)
#                 improvement = True

#         # Check for convergence
#         current_cut_size = nx.cut_size(graph, partition_a, partition_b)
#         if current_cut_size >= initial_cut_size or not improvement:
#             break

#     return partition_a, partition_b



def kargerMinCut(graph):
    # Get data of given graph
    V = graph.V
    E = graph.E
    edge = graph.edge

    # Allocate memory for creating V subsets.
    subsets = []

    # Create V subsets with single elements
    for v in range(V):
        subsets.append(subset(v, 0))

    # Initially there are V vertices in
    # contracted graph
    vertices = V

    # Keep contracting vertices until there are
    # 2 vertices.
    while vertices > 2:
        # Pick a random edge
        i = random.randint(0, E - 1)

        # Find vertices (or sets) of two corners
        # of the current edge
        subset1 = find(subsets, edge[i].src)
        subset2 = find(subsets, edge[i].dest)

        # If two corners belong to the same subset,
        # then no point considering this edge
        if subset1 == subset2:
            continue

        # Else contract the edge (or combine the
        # corners of edge into one vertex)
        else:
            print("Contracting edge " + str(edge[i].src) + "-" + str(edge[i].dest))
            vertices -= 1
            Union(subsets, subset1, subset2)

        # Draw the contracted graph
        contracted_G = nx.Graph()
        for i in range(E):
            subset1 = find(subsets, edge[i].src)
            subset2 = find(subsets, edge[i].dest)
            if subset1 != subset2:
                contracted_G.add_edge(subset1, subset2)

        # Draw the original graph with a colormap
        pos = nx.spring_layout(contracted_G)
        nx.draw(contracted_G, pos=pos, with_labels=True, font_weight='bold', node_color="green", edge_color="black")
        plt.show()

    # Now we have two vertices (or subsets) left in
    # the contracted graph, so count the edges between
    # two components and return the count.
    cut_edges = 0
    for i in range(E):
        subset1 = find(subsets, edge[i].src)
        subset2 = find(subsets, edge[i].dest)
        if subset1 != subset2:
            cut_edges += 1

    # Construct the updated graph after contraction
    updated_G = nx.Graph()
    for i in range(E):
        subset1 = find(subsets, edge[i].src)
        subset2 = find(subsets, edge[i].dest)
        if subset1 != subset2:
            updated_G.add_edge(subset1, subset2)

    return cut_edges, updated_G


def kernighan_lin(graph):
    total_nodes = len(graph.nodes())
    partition_a = set(random.sample(graph.nodes(), total_nodes // 2))
    partition_b = set(graph.nodes()) - partition_a

    # Calculate initial cut size
    initial_cut_size = nx.cut_size(graph, partition_a, partition_b)

    while True:
        improvement = False

        # Iterate over vertices and try swapping between partitions
        for node in partition_a:
            gain = sum(1 for neighbor in graph.neighbors(node) if neighbor in partition_b) - \
                   sum(1 for neighbor in graph.neighbors(node) if neighbor in partition_a)

            # Find a node in partition_b to swap with
            swap_candidate = max(partition_b, key=lambda x: len(set(graph.neighbors(x)) & partition_a))

            if gain > 0:
                partition_a.remove(node)
                partition_b.remove(swap_candidate)
                partition_a.add(swap_candidate)
                partition_b.add(node)
                improvement = True

        # Check for convergence
        current_cut_size = nx.cut_size(graph, partition_a, partition_b)
        if current_cut_size >= initial_cut_size or not improvement:
            break

    return partition_a, partition_b
def louvain_partition(graph):
    partition = community.best_partition(graph)
    partitions = {}

    # Group nodes by partition
    for node, part in partition.items():
        if part not in partitions:
            partitions[part] = set()
        partitions[part].add(node)

    return partitions

# def compare_algorithms(num_nodes, num_trials):
#     print(f"\nComparing algorithms for {num_nodes} nodes:")
#     min_cut_kg = float('inf')
#     min_cut_kl = float('inf')
#     min_cut_lo = float('inf')
#     best_algorithm = ''

#     for _ in range(num_trials):
#         # Generate a random graph
#         G = nx.erdos_renyi_graph(num_nodes, 0.2)

#         # Convert the graph to the format required for Karger's algorithm
#         graph = Graph(num_nodes, len(G.edges()))
#         for edge in G.edges():
#             graph.edge.append(Edge(edge[0], edge[1]))

#         # Measure the cut size for Karger's algorithm
#         cut_edges, _ = kargerMinCut(graph)
#         min_cut_kg = min(min_cut_kg, cut_edges)

#         # Measure the cut size for Kernighan-Lin algorithm
#         partition_a, partition_b = kernighan_lin(G)
#         cut_edges_kl = nx.cut_size(G, partition_a, partition_b)
#         min_cut_kl = min(min_cut_kl, cut_edges_kl)

#         # Measure the cut size for Louvain method
#         partition = louvain_partition(copy.deepcopy(G))
#         cut_edges_lo = nx.cut_size(G, partition[0], partition[1])
#         min_cut_lo = min(min_cut_lo, cut_edges_lo)

#     if min_cut_kg <= min_cut_kl and min_cut_kg <= min_cut_lo:
#         best_algorithm = "Karger's Algorithm"
#     elif min_cut_kl <= min_cut_kg and min_cut_kl <= min_cut_lo:
#         best_algorithm = "Kernighan-Lin Algorithm"
#     else:
#         best_algorithm = "Louvain Method"

#     print(f"The best algorithm for {num_nodes} nodes is {best_algorithm} with a minimum cut of {min(min_cut_kg, min_cut_kl, min_cut_lo)}")
def main():
    # Create a graph
    G = nx.Graph()
    nodes = int(input("How many nodes?: "))

    for i in range(nodes):
        G.add_edge(i, (i + 1) % nodes)

    # Convert the graph to the format required for Karger's algorithm
    graph = Graph(nodes, len(G.edges()))
    for edge in G.edges():
        graph.edge.append(Edge(edge[0], edge[1]))

    # Provide an option to run Karger's algorithm
    while True:
        visualize_option = input("\nType (kg/kl/lo/exit): ")

        if visualize_option.lower() == "exit":
            sys.exit("Exiting the program.")

        elif visualize_option.lower() == "kg":
            cut_edges, updated_G = kargerMinCut(graph)
            print(f"\nCut found by Karger's randomized algo is {cut_edges}")

            # Draw the updated graph
            pos = nx.spring_layout(updated_G)
            nx.draw(updated_G, pos=pos, with_labels=True, font_weight='bold', node_color="green", edge_color="black")
            plt.show()
        elif visualize_option.lower() == "lo":
            partitions = louvain_partition(G)

            fig, axs = plt.subplots(1, len(partitions), figsize=(15, 5))

            for i, (part, nodes_in_part) in enumerate(partitions.items()):
                subgraph = G.subgraph(nodes_in_part)
                pos = nx.spring_layout(subgraph)
                nx.draw(subgraph, pos=pos, with_labels=True, font_weight='bold', node_color="green", edge_color="black", ax=axs[i])
                axs[i].set_title(f"Partition {part}")
            plt.show()

        elif visualize_option.lower() == "kl":
            partition_a, partition_b = kernighan_lin(G)

            print("\nFinal Partition A:", partition_a)
            print("Final Partition B:", partition_b)

            # Create a subgraph for each partition
            subgraph_a = G.subgraph(partition_a)
            subgraph_b = G.subgraph(partition_b)

            # Draw each subgraph separately
            pos_a = nx.spring_layout(subgraph_a)
            pos_b = nx.spring_layout(subgraph_b)

            plt.subplot(121)
            nx.draw(subgraph_a, pos=pos_a, with_labels=True, font_weight='bold', node_color="green", edge_color="black")
            plt.title("Partition A")

            plt.subplot(122)
            nx.draw(subgraph_b, pos=pos_b, with_labels=True, font_weight='bold', node_color="red", edge_color="black")
            plt.title("Partition B")
            plt.show()

        else:
            print("Invalid input.")

if __name__=='__main__':
    main()
