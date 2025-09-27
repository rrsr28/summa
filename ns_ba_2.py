import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import powerlaw
import random

# ================= Utility Functions =================
def plot_degree_distribution(G, label="", cumulative=False):
    """Plot degree distribution on log-log scale."""
    degrees = [d for n, d in G.degree()]
    hist, bins = np.histogram(degrees, bins=range(min(degrees), max(degrees)+2))
    if cumulative:
        hist = np.cumsum(hist[::-1])[::-1] / len(degrees)
    else:
        hist = hist / len(degrees)
    plt.loglog(bins[:-1], hist, marker="o", linestyle="", label=label)

def compute_centralities(G):
    """Compute centrality measures for a simple graph."""
    return {
        "degree": nx.degree_centrality(G),
        "betweenness": nx.betweenness_centrality(G),
        "closeness": nx.closeness_centrality(G),
        "eigenvector": nx.eigenvector_centrality(G, max_iter=500),
        "pagerank": nx.pagerank(G),
    }

def clean_graph(G):
    """Convert MultiGraph/MultiDiGraph to simple Graph/DiGraph and remove self-loops."""
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        # Collapse parallel edges
        if G.is_directed():
            G = nx.DiGraph(G)
        else:
            G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

# ================= File Loader =================
def load_graph_from_file(filename, directed=False, filetype="gml"):
    """
    Load a graph from a file.
    :param filename: path to file
    :param directed: True for directed, False for undirected
    :param filetype: "gml", "edgelist", or "adjlist"
    """
    if filetype == "gml":
        G = nx.read_gml(filename)
    elif filetype == "edgelist":
        if directed:
            G = nx.read_edgelist(filename, create_using=nx.DiGraph, nodetype=int)
        else:
            G = nx.read_edgelist(filename, create_using=nx.Graph, nodetype=int)
    elif filetype == "adjlist":
        if directed:
            G = nx.read_adjlist(filename, create_using=nx.DiGraph, nodetype=int)
        else:
            G = nx.read_adjlist(filename, create_using=nx.Graph, nodetype=int)
    else:
        raise ValueError("Unsupported filetype. Use gml, edgelist, or adjlist.")

    return clean_graph(G)

# ================= Main Analysis =================
if __name__ == "__main__":
    # ----- Option 1: Generate BA networks and analyze growth -----
    sizes = [10**2, 10**3, 10**4]
    ba_networks = {N: nx.barabasi_albert_graph(N, m=3) for N in sizes}

    plt.figure(figsize=(8,6))
    for N, G in ba_networks.items():
        plot_degree_distribution(G, label=f"BA N={N}")
    plt.xlabel("Degree (k)")
    plt.ylabel("P(k)")
    plt.legend()
    plt.title("Degree Distribution of BA Networks")
    plt.show()

    for N, G in ba_networks.items():
        degrees = [d for n, d in G.degree()]
        fit = powerlaw.Fit(degrees, discrete=True, verbose=False)
        print(f"BA N={N}: exponent gamma ~ {fit.power_law.alpha:.2f}")

    plt.figure(figsize=(8,6))
    for N, G in ba_networks.items():
        plot_degree_distribution(G, label=f"BA N={N}", cumulative=True)
    plt.xlabel("Degree (k)")
    plt.ylabel("Cumulative P(k)")
    plt.legend()
    plt.title("Cumulative Degree Distributions")
    plt.show()

    Ns = np.logspace(2, 4, 10, dtype=int)
    clustering = []
    for N in Ns:
        G = nx.barabasi_albert_graph(N, 3)
        clustering.append(nx.average_clustering(G))

    plt.figure(figsize=(8,6))
    plt.plot(Ns, clustering, marker="o")
    plt.xscale("log")
    plt.xlabel("Number of nodes (N)")
    plt.ylabel("Average clustering coefficient")
    plt.title("Clustering coefficient vs N (BA)")
    plt.show()

    # ----- Option 2: Model comparisons -----
    N, m = 1000, 3

    # Generate degree sequences ensuring even sums
    uniform_seq = [random.randint(1, 10) for _ in range(N)]
    if sum(uniform_seq) % 2 != 0:
        uniform_seq[-1] += 1
    normal_seq = [max(1, int(random.gauss(5, 2))) for _ in range(N)]
    if sum(normal_seq) % 2 != 0:
        normal_seq[-1] += 1

    models = {
        "BA": nx.barabasi_albert_graph(N, m),
        "ER": nx.erdos_renyi_graph(N, p=0.01),
        "Power-law cluster": nx.powerlaw_cluster_graph(N, m, 0.3),
        "Watts-Strogatz": nx.watts_strogatz_graph(N, k=6, p=0.3),
        "Uniform degree": nx.configuration_model(uniform_seq),
        "Normal degree": nx.configuration_model(normal_seq),
        # Uncomment to load from file:
        # "File graph": load_graph_from_file("your_network.gml", directed=False, filetype="gml"),
    }

    for name in models:
        models[name] = clean_graph(models[name])

    plt.figure(figsize=(8,6))
    for name, G in models.items():
        plot_degree_distribution(G, label=name)
    plt.xlabel("Degree (k)")
    plt.ylabel("P(k)")
    plt.legend()
    plt.title("Degree Distributions: Model Comparison")
    plt.show()

    plt.figure(figsize=(8,6))
    for name, G in models.items():
        plot_degree_distribution(G, label=name, cumulative=True)
    plt.xlabel("Degree (k)")
    plt.ylabel("Cumulative P(k)")
    plt.legend()
    plt.title("Cumulative Degree Distributions: Model Comparison")
    plt.show()

    centrality_measures = {name: compute_centralities(G) for name, G in models.items()}

    print("\nTop 5 nodes in BA model by degree centrality:")
    ba_deg = centrality_measures["BA"]["degree"]
    print(sorted(ba_deg.items(), key=lambda x: x[1], reverse=True)[:5])
