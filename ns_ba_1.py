import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import powerlaw
import random
from networkx.algorithms.community import greedy_modularity_communities

# ===================== Utility Functions =====================
def plot_degree_distribution(G, label="", cumulative=False):
    degrees = [d for n, d in G.degree()]
    hist, bins = np.histogram(degrees, bins=range(min(degrees), max(degrees)+2))
    if cumulative:
        hist = np.cumsum(hist[::-1])[::-1] / len(degrees)
    else:
        hist = hist / len(degrees)
    plt.loglog(bins[:-1], hist, marker="o", linestyle="", label=label)

def compute_centralities(G):
    return {
        "degree": nx.degree_centrality(G),
        "betweenness": nx.betweenness_centrality(G),
        "closeness": nx.closeness_centrality(G),
        "eigenvector": nx.eigenvector_centrality(G, max_iter=500),
        "pagerank": nx.pagerank(G),
    }

def clean_graph(G):
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def percolation(G, fraction, targeted=False):
    G_copy = G.copy()
    if targeted:
        nodes_to_remove = sorted(G_copy.degree, key=lambda x: x[1], reverse=True)
        nodes_to_remove = [n for n, d in nodes_to_remove]
    else:
        nodes_to_remove = list(G_copy.nodes())
        np.random.shuffle(nodes_to_remove)
    num_remove = int(fraction * len(G_copy))
    G_copy.remove_nodes_from(nodes_to_remove[:num_remove])
    components = list(nx.connected_components(G_copy))
    return len(max(components, key=len, default=set()))

def sir_simulation(G, beta=0.1, gamma=0.05, steps=100):
    status = {n: 'S' for n in G}
    status[0] = 'I'
    infected_count = []
    for _ in range(steps):
        new_status = status.copy()
        for n in G:
            if status[n] == 'I':
                for neighbor in G.neighbors(n):
                    if status[neighbor] == 'S' and np.random.rand() < beta:
                        new_status[neighbor] = 'I'
                if np.random.rand() < gamma:
                    new_status[n] = 'R'
        status = new_status
        infected_count.append(sum(1 for s in status.values() if s == 'I'))
    return max(infected_count)

def plot_network(G, original_pos=None, removed_nodes=None, title="Network"):
    if original_pos is None:
        original_pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(6,6))
    nx.draw_networkx_nodes(G, original_pos, node_color="skyblue", node_size=500, alpha=0.8)
    nx.draw_networkx_edges(G, original_pos, alpha=0.5)
    nx.draw_networkx_labels(G, original_pos, font_size=10, font_color="black")
    if removed_nodes:
        nx.draw_networkx_nodes(G, original_pos, nodelist=removed_nodes,
                               node_color="red", node_size=600, alpha=0.8, label="Removed")
    plt.title(title)
    plt.axis("off")
    plt.show()

# ===================== Step 1: BA Network =====================
n = 1000
m = 3
G = nx.barabasi_albert_graph(n, m)

# Basic metrics
avg_path_length = nx.average_shortest_path_length(G)
avg_clustering = nx.average_clustering(G)
diameter = nx.diameter(G)
assortativity = nx.degree_assortativity_coefficient(G)

degrees = [d for _, d in G.degree()]
min_degree = min(degrees)
max_degree = max(degrees)
avg_degree = np.mean(degrees)

# Power-law fit
fit = powerlaw.Fit(degrees, verbose=False)
power_law_alpha = fit.power_law.alpha
power_law_xmin = fit.power_law.xmin

# Centrality
betweenness = nx.betweenness_centrality(G)
top_hubs = sorted(betweenness, key=betweenness.get, reverse=True)[:5]

# Robustness
largest_component_random = percolation(G, 0.2, targeted=False)
largest_component_targeted = percolation(G, 0.2, targeted=True)

# SIR simulation
max_infected = sir_simulation(G)

# Community detection
communities = greedy_modularity_communities(G)
num_communities = len(communities)
modularity = nx.community.modularity(G, communities)

# ===================== Step 2: Plots =====================
# Degree distribution
hist, bin_edges = np.histogram(degrees, bins=50, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
valid = hist > 0

plt.figure(figsize=(8,6))
plt.loglog(bin_centers[valid], hist[valid], 'b.', label='Degree distribution')
plt.xlabel('Degree (k)')
plt.ylabel('P(k)')
plt.title('Degree Distribution of BA Network')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

# Network visualization
plt.figure(figsize=(8,8))
pos = nx.spring_layout(G)
node_sizes = [d * 10 for _, d in G.degree()]
nx.draw(G, pos, node_size=node_sizes, node_color='skyblue', edge_color='gray', alpha=0.6)
plt.title('Barabási-Albert Network Visualization')
plt.show()

# ===================== Step 3: BA Growth Analysis =====================
sizes = [10**2, 10**3, 10**4]
ba_networks = {N: nx.barabasi_albert_graph(N, m=3) for N in sizes}

plt.figure(figsize=(8,6))
for N, G_tmp in ba_networks.items():
    plot_degree_distribution(G_tmp, label=f"BA N={N}")
plt.xlabel("Degree (k)")
plt.ylabel("P(k)")
plt.legend()
plt.title("Degree Distribution of BA Networks")
plt.show()

# Power-law exponents
for N, G_tmp in ba_networks.items():
    degrees_tmp = [d for _, d in G_tmp.degree()]
    fit_tmp = powerlaw.Fit(degrees_tmp, discrete=True, verbose=False)
    print(f"BA N={N}: exponent gamma ~ {fit_tmp.power_law.alpha:.2f}")

# Cumulative distributions
plt.figure(figsize=(8,6))
for N, G_tmp in ba_networks.items():
    plot_degree_distribution(G_tmp, label=f"BA N={N}", cumulative=True)
plt.xlabel("Degree (k)")
plt.ylabel("Cumulative P(k)")
plt.legend()
plt.title("Cumulative Degree Distributions")
plt.show()

# Clustering vs N
Ns = np.logspace(2,4,10, dtype=int)
clustering = []
for N_tmp in Ns:
    G_tmp = nx.barabasi_albert_graph(N_tmp, m)
    clustering.append(nx.average_clustering(G_tmp))

plt.figure(figsize=(8,6))
plt.plot(Ns, clustering, marker="o")
plt.xscale("log")
plt.xlabel("Number of nodes (N)")
plt.ylabel("Average clustering coefficient")
plt.title("Clustering coefficient vs N (BA)")
plt.show()

# ===================== Step 4: Model Comparisons =====================
N = 1000
m = 3
# Ensure even degree sequences
uniform_seq = [random.randint(1,10) for _ in range(N)]
if sum(uniform_seq) % 2 != 0: uniform_seq[-1] += 1
normal_seq = [max(1,int(random.gauss(5,2))) for _ in range(N)]
if sum(normal_seq) % 2 != 0: normal_seq[-1] += 1

models = {
    "BA": nx.barabasi_albert_graph(N, m),
    "ER": nx.erdos_renyi_graph(N, p=0.01),
    "Power-law cluster": nx.powerlaw_cluster_graph(N, m, 0.3),
    "Watts-Strogatz": nx.watts_strogatz_graph(N, k=6, p=0.3),
    "Uniform degree": nx.configuration_model(uniform_seq),
    "Normal degree": nx.configuration_model(normal_seq)
}

# Clean graphs
for key in models:
    models[key] = clean_graph(models[key])

# Degree distribution comparison
plt.figure(figsize=(8,6))
for name, G_tmp in models.items():
    plot_degree_distribution(G_tmp, label=name)
plt.xlabel("Degree (k)")
plt.ylabel("P(k)")
plt.legend()
plt.title("Degree Distributions: Model Comparison")
plt.show()

# Cumulative distributions comparison
plt.figure(figsize=(8,6))
for name, G_tmp in models.items():
    plot_degree_distribution(G_tmp, label=name, cumulative=True)
plt.xlabel("Degree (k)")
plt.ylabel("Cumulative P(k)")
plt.legend()
plt.title("Cumulative Degree Distributions: Model Comparison")
plt.show()

# Centralities
centrality_measures = {name: compute_centralities(G_tmp) for name, G_tmp in models.items()}

print("\nTop 5 nodes in BA model by degree centrality:")
ba_deg = centrality_measures["BA"]["degree"]
print(sorted(ba_deg.items(), key=lambda x: x[1], reverse=True)[:5])

# ===================== Step 5: Print Summary =====================
print("\n=== Barabási-Albert Network Metrics ===")
print(f"Avg Shortest Path: {avg_path_length:.3f}")
print(f"Avg Clustering: {avg_clustering:.3f}")
print(f"Diameter: {diameter}")
print(f"Degree Assortativity: {assortativity:.3f}")
print(f"Min Degree: {min_degree}, Max Degree: {max_degree}, Avg Degree: {avg_degree:.3f}")
print(f"Power-law alpha: {power_law_alpha:.3f}, xmin: {power_law_xmin}")
print(f"Top 5 Hubs (betweenness): {top_hubs}")
print(f"Largest Component (20% random removal): {largest_component_random}")
print(f"Largest Component (20% targeted removal): {largest_component_targeted}")
print(f"Max Infected in SIR: {max_infected}")
print(f"Number of communities: {num_communities}, Modularity: {modularity:.3f}")
