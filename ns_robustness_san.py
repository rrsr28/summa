import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from powerlaw import Fit
import random
from networkx_robustness import networkx_robustness as netrob

# ===================== Hietatchical  =====================
def hierarchical_model(level=3,m=4):
  g= nx.complete_graph(m)

  for l in range(1,level):
    for node in list(g.nodes()):
      module = nx.complete_graph(m)
      mapping = {i:f"{node}_{l}_{i}" for i in list(module.nodes())}
      module = nx.relabel_nodes(module,mapping)
      g = nx.union(g,module)
      g.add_edge(node, f"{node}_{l}_0")

  return g

g = hierarchical_model()

plt.figure(figsize=(10,10))
pos = nx.spring_layout(g,seed =42)
nx.draw(g,pos,node_size=50,node_color='skyblue')
plt.show()

# ===================== Utility Functions =====================
def compute_metrics_and_distribution(G, label="Network"):
    #Compute network metrics, degree distribution, and power-law fit.
    metrics = {}
    if nx.is_connected(G):
        metrics['avg_path_length'] = nx.average_shortest_path_length(G)
        metrics['diameter'] = nx.diameter(G)
    else:
        metrics['avg_path_length'] = float('inf')
        metrics['diameter'] = float('inf')

    metrics['avg_clustering'] = nx.average_clustering(G)
    metrics['assortativity'] = nx.degree_assortativity_coefficient(G)

    degrees = [d for _, d in G.degree()]
    metrics['min_degree'] = min(degrees) if degrees else 0
    metrics['max_degree'] = max(degrees) if degrees else 0
    metrics['avg_degree'] = np.mean(degrees) if degrees else 0

    # Degree distribution
    hist, bin_edges = np.histogram(degrees, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    valid = hist > 0

    # Power-law fit
    power_law_stats = {}
    fit = None
    if degrees:
        fit = Fit(degrees, verbose=False)
        power_law_stats['alpha'] = fit.power_law.alpha
        power_law_stats['xmin'] = fit.power_law.xmin
        R, p = fit.distribution_compare('power_law', 'exponential')
        power_law_stats['vs_exponential_R'] = R
        power_law_stats['vs_exponential_p'] = p
    else:
        power_law_stats['alpha'] = power_law_stats['xmin'] = 0
        power_law_stats['vs_exponential_R'] = 0
        power_law_stats['vs_exponential_p'] = 1

    return metrics, (hist[valid], bin_centers[valid], fit, power_law_stats)

def remove_nodes(G, fraction, removal_type='random'):
    #Remove fraction of nodes randomly or targeted (by degree).
    G_copy = G.copy()
    N = len(G_copy)
    num_remove = int(fraction * N)
    if removal_type == 'targeted':
        nodes_to_remove = sorted(G_copy.degree, key=lambda x: x[1], reverse=True)
        nodes_to_remove = [node[0] for node in nodes_to_remove[:num_remove]]
    else:
        nodes_to_remove = list(G_copy.nodes())
        random.shuffle(nodes_to_remove)
        nodes_to_remove = nodes_to_remove[:num_remove]
    G_copy.remove_nodes_from(nodes_to_remove)
    return G_copy

def percolation_simulation(G, removal_type='random', steps=50):
    #Simulate percolation and return largest component fraction at each step.
    N = len(G)
    fractions = np.linspace(0, 1, steps)
    largest_components = []

    for f in fractions:
        G_copy = remove_nodes(G, f, removal_type)
        if len(G_copy) == 0:
            largest_components.append(0)
        else:
            components = nx.connected_components(G_copy)
            largest = max(len(c) for c in components) / N
            largest_components.append(largest)

    return fractions, largest_components

def find_failure_threshold(fractions, largest_components, threshold=0.01):
    #Estimate percolation threshold where largest component <= threshold.
    for i, size in enumerate(largest_components):
        if size <= threshold:
            return fractions[i]
    return 1.0

def plot_degree_distribution(hist_bins_list, labels, title="Degree Distribution"):
    plt.figure(figsize=(10,5))
    for (hist, bins), label in zip(hist_bins_list, labels):
        plt.loglog(bins, hist, marker='o', linestyle='', label=label)
    plt.xlabel("Degree (k)")
    plt.ylabel("P(k)")
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.show()

def generate_networks(n=1000, p_er=0.006, m_ba=3, k_ws=6, p_ws=0.1):
    #Generate ER, BA, and WS networks.
    ER = nx.erdos_renyi_graph(n, p_er)
    BA = nx.barabasi_albert_graph(n, m_ba)
    WS = nx.watts_strogatz_graph(n, k_ws, p_ws)
    return ER, BA, WS

# ===================== Main Analysis =====================
if __name__ == "__main__":
    print("=== Complex Network Robustness and Scale-Free Analysis ===")
    n = 100
    m_ba = 3
    p_er = 2*m_ba/(n-1)  # ER probability to match BA average degree ~6
    fraction_remove = 0.2  # 20% nodes removed

    # Generate networks
    ER, BA, WS = generate_networks(n=n, p_er=p_er, m_ba=m_ba)
    print("Networks generated: ER, BA, WS")

    # ===================== Network Robustness Attack Simulations =====================
    print("\n=== Network Attack Simulations ===")
    
    # Test on Barabási-Albert network
    G = BA
    print(f"Testing attacks on BA network with {len(G)} nodes")
    
    # Attack simulations using networkx_robustness
    initial, frac_random, apl_random = netrob.simulate_random_attack(G, attack_fraction=0.2)
    initial, frac_degree, apl_degree = netrob.simulate_degree_attack(G, attack_fraction=0.1, weight=None)
    initial, frac_betweenness, apl_betweenness = netrob.simulate_betweenness_attack(G, attack_fraction=0.1, weight=None)
    initial, frac_closeness, apl_closeness = netrob.simulate_closeness_attack(G, attack_fraction=0.1, weight=None)
    initial, frac_eigenvector, apl_eigenvector = netrob.simulate_eigenvector_attack(G, attack_fraction=0.1, weight=None)

    print(f"Attack simulations completed - Initial APL: {initial:.3f}")

    # Plot attack results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(frac_random, apl_random, 'b-', label='Random Attack', linewidth=2)
    plt.xlabel('Fraction of Nodes Removed')
    plt.ylabel('Average Path Length')
    plt.title('Random Attack on BA Network')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(frac_degree, apl_degree, 'r-', label='Degree Attack', linewidth=2)
    plt.plot(frac_betweenness, apl_betweenness, 'g-', label='Betweenness Attack', linewidth=2)
    plt.xlabel('Fraction of Nodes Removed')
    plt.ylabel('Average Path Length')
    plt.title('Centrality-Based Attacks on BA Network')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(frac_closeness, apl_closeness, 'purple', label='Closeness Attack', linewidth=2)
    plt.plot(frac_eigenvector, apl_eigenvector, 'orange', label='Eigenvector Attack', linewidth=2)
    plt.xlabel('Fraction of Nodes Removed')
    plt.ylabel('Average Path Length')
    plt.title('Advanced Centrality Attacks on BA Network')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(frac_random, apl_random, 'b-', label='Random', linewidth=2)
    plt.plot(frac_degree, apl_degree, 'r-', label='Degree', linewidth=2)
    plt.plot(frac_betweenness, apl_betweenness, 'g-', label='Betweenness', linewidth=2)
    plt.plot(frac_closeness, apl_closeness, 'purple', label='Closeness', linewidth=2)
    plt.plot(frac_eigenvector, apl_eigenvector, 'orange', label='Eigenvector', linewidth=2)
    plt.xlabel('Fraction of Nodes Removed')
    plt.ylabel('Average Path Length')
    plt.title('All Attack Strategies Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()

    # ===================== Original Analysis =====================
    # Compute metrics before removal
    ba_metrics, ba_data = compute_metrics_and_distribution(BA, "BA Original")
    er_metrics, er_data = compute_metrics_and_distribution(ER, "ER Original")

    # Node removal: Random and Targeted
    BA_rand = remove_nodes(BA, fraction_remove, 'random')
    BA_targ = remove_nodes(BA, fraction_remove, 'targeted')
    ER_rand = remove_nodes(ER, fraction_remove, 'random')
    ER_targ = remove_nodes(ER, fraction_remove, 'targeted')

    # Compute metrics after removal
    ba_rand_metrics, ba_rand_data = compute_metrics_and_distribution(BA_rand, "BA Random Removal")
    ba_targ_metrics, ba_targ_data = compute_metrics_and_distribution(BA_targ, "BA Targeted Removal")
    er_rand_metrics, er_rand_data = compute_metrics_and_distribution(ER_rand, "ER Random Removal")
    er_targ_metrics, er_targ_data = compute_metrics_and_distribution(ER_targ, "ER Targeted Removal")

    # Print summary metrics
    print("\nBA Original Metrics:", ba_metrics)
    print("BA 20% Random Removal Metrics:", ba_rand_metrics)
    print("BA 20% Targeted Removal Metrics:", ba_targ_metrics)
    print("\nER Original Metrics:", er_metrics)
    print("ER 20% Random Removal Metrics:", er_rand_metrics)
    print("ER 20% Targeted Removal Metrics:", er_targ_metrics)

    # Percolation simulations
    nets = {'ER': ER, 'BA': BA, 'WS': WS}
    fig, axs = plt.subplots(1,2, figsize=(12,5))
    for name, G_net in nets.items():
        fr_rand, lc_rand = percolation_simulation(G_net, 'random')
        fr_targ, lc_targ = percolation_simulation(G_net, 'targeted')
        fc_rand = find_failure_threshold(fr_rand, lc_rand)
        fc_targ = find_failure_threshold(fr_targ, lc_targ)
        axs[0].plot(fr_rand, lc_rand, label=f"{name} (fc={fc_rand:.2f})")
        axs[1].plot(fr_targ, lc_targ, label=f"{name} (fc={fc_targ:.2f})")

    axs[0].set_title('Random Node Removal')
    axs[0].set_xlabel('Fraction Removed')
    axs[0].set_ylabel('Normalized Largest Component')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_title('Targeted Node Removal')
    axs[1].set_xlabel('Fraction Removed')
    axs[1].set_ylabel('Normalized Largest Component')
    axs[1].legend()
    axs[1].grid(True)
    plt.tight_layout()
    plt.show()

    # Degree distributions plots
    plot_degree_distribution(
        [(ba_data[0], ba_data[1]), (ba_rand_data[0], ba_rand_data[1]), (ba_targ_data[0], ba_targ_data[1])],
        labels=["BA Original", "BA Random 20%", "BA Targeted 20%"],
        title="BA Degree Distribution"
    )
    plot_degree_distribution(
        [(er_data[0], er_data[1]), (er_rand_data[0], er_rand_data[1]), (er_targ_data[0], er_targ_data[1])],
        labels=["ER Original", "ER Random 20%", "ER Targeted 20%"],
        title="ER Degree Distribution"
    )

    # Network visualizations
    plt.figure(figsize=(12,10))
    pos = nx.spring_layout(BA, seed=42)
    plt.subplot(2,2,1)
    nx.draw(BA, pos, node_size=[d*10 for _,d in BA.degree()], node_color='skyblue', edge_color='gray', alpha=0.6)
    plt.title('BA Original')

    plt.subplot(2,2,2)
    pos = nx.spring_layout(BA_targ, seed=42)
    nx.draw(BA_targ, pos, node_size=[d*10 for _,d in BA_targ.degree()], node_color='red', edge_color='gray', alpha=0.6)
    plt.title('BA 20% Targeted Removal')

    plt.subplot(2,2,3)
    pos = nx.spring_layout(ER, seed=42)
    nx.draw(ER, pos, node_size=[d*10 for _,d in ER.degree()], node_color='skyblue', edge_color='gray', alpha=0.6)
    plt.title('ER Original')

    plt.subplot(2,2,4)
    pos = nx.spring_layout(ER_targ, seed=42)
    nx.draw(ER_targ, pos, node_size=[d*10 for _,d in ER_targ.degree()], node_color='red', edge_color='gray', alpha=0.6)
    plt.title('ER 20% Targeted Removal')

    plt.tight_layout()
    plt.show()





#===================================================BA Network ===================================================
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from powerlaw import Fit

# --- Generate BA Network ---
n, m = 1000, 3  # n = nodes, m = edges per new node
BA = nx.barabasi_albert_graph(n, m)

print("Nodes:", BA.number_of_nodes())
print("Edges:", BA.number_of_edges())

# --- Degree distribution ---
degrees = [d for _, d in BA.degree()]
plt.hist(degrees, bins=30, density=True, alpha=0.7, color="skyblue")
plt.xlabel("Degree (k)")
plt.ylabel("P(k)")
plt.title("Degree Distribution of BA Network")
plt.show()

# --- Power law test ---
fit = Fit(degrees, verbose=False)
print(f"Power law alpha (γ): {fit.power_law.alpha:.2f}")
print(f"xmin (cutoff): {fit.power_law.xmin}")
R, p = fit.distribution_compare("power_law", "exponential")
print(f"Power-law vs Exponential: R={R:.2f}, p={p:.3f}")

# --- Preferential attachment check ---
# Sort degrees descending (node rank vs degree)
deg_sorted = sorted(degrees, reverse=True)
plt.plot(range(1, len(deg_sorted)+1), deg_sorted, "bo")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Node Rank (log scale)")
plt.ylabel("Degree (log scale)")
plt.title("Preferential Attachment (Rich get richer)")
plt.show()

# --- Growth property ---
print("Is graph connected?", nx.is_connected(BA))
print("Average degree:", np.mean(degrees))
print("Average clustering coefficient:", nx.average_clustering(BA))
print("Diameter:", nx.diameter(BA) if nx.is_connected(BA) else "Not connected")

# --- Visualize the network (small sample) ---
plt.figure(figsize=(7,7))
sample = nx.barabasi_albert_graph(50, 2)  # small network to see structure
pos = nx.spring_layout(sample, seed=42)
nx.draw(sample, pos, node_size=[d*30 for _, d in sample.degree()],
        node_color="skyblue", edge_color="gray", with_labels=False)
plt.title("Sample BA Network (n=50, m=2)")
plt.show()





#============================================Chat hierarchical ============================================================
"""
Modified to:
 - ensure the degree sequence sum is even (avoids NetworkXError)
 - provide a controlled hierarchical/modular network generator
 - minor performance improvements in attack simulation
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

# ===== Utility Functions =====
def remove_nodes_by_list(G, nodes_to_remove):
    """Return graph with given nodes removed (works on a copy)."""
    Gc = G.copy()
    Gc.remove_nodes_from(nodes_to_remove)
    return Gc

def giant_component_fraction(G):
    """Return fraction of nodes in largest connected component"""
    if len(G) == 0:
        return 0.0
    return max(len(c) for c in nx.connected_components(G)) / len(G)

def simulate_attack(G, metric='degree', steps=50):
    """
    Simulate targeted attack using a static ranking (computed once).
    This is much faster than recomputing metrics after each removal,
    and is a reasonable approximation for targeted attacks.
    """
    N = len(G)
    fractions = np.linspace(0, 1, steps)
    giant_sizes = []
    # Compute ranking once
    if metric == 'degree':
        ranking = sorted(G.degree, key=lambda x: x[1], reverse=True)
        ranked_nodes = [n for n, _ in ranking]
    elif metric == 'clustering':
        clustering = nx.clustering(G)
        ranking = sorted(clustering.items(), key=lambda x: x[1], reverse=True)
        ranked_nodes = [n for n, _ in ranking]
    else:
        raise ValueError("metric must be 'degree' or 'clustering'")

    for f in fractions:
        num_remove = int(f * N)
        nodes_to_remove = ranked_nodes[:num_remove]
        Gc = remove_nodes_by_list(G, nodes_to_remove)
        giant_sizes.append(giant_component_fraction(Gc))
    return fractions, giant_sizes

# ===== Generate Networks =====
N = 10000  # target number of nodes for the configuration model (may be large)

# 1. Configuration model network with power-law degree distribution (gamma=2.5)
gamma = 2.5
# generate power-law floats then convert to ints >=1
degrees = nx.utils.powerlaw_sequence(N, exponent=gamma)
degrees = [int(max(1, d)) for d in degrees]

# Ensure sum of degrees is even (necessary for configuration_model)
if sum(degrees) % 2 != 0:
    # adjust a random degree by +1 (keeps degrees >=1)
    idx = random.randrange(len(degrees))
    degrees[idx] += 1

# sanity check
assert sum(degrees) % 2 == 0, "Degree sum must be even"

# create configuration model and simplify (remove parallel edges and self-loops)
G_config = nx.configuration_model(degrees, create_using=None)
G_config = nx.Graph(G_config)  # collapses parallel edges
G_config.remove_edges_from(nx.selfloop_edges(G_config))
print(f"Configuration model: nodes={G_config.number_of_nodes()}, edges={G_config.number_of_edges()}")

# 2. Controlled hierarchical / modular network
def hierarchical_modular_network(levels=3, module_size=8, branching=2):
    """
    Build a hierarchical modular network:
      - level 1: create branching modules each a complete graph of size module_size
      - to build next level, create branching copies of the current structure and connect their module hubs
    This scales as: nodes = module_size * (branching ** (levels - 1))
    and is controlled by the branching factor to avoid explosive growth.
    """
    if levels < 1:
        raise ValueError("levels must be >= 1")

    # Start with a single module (complete graph of module_size)
    base_module = nx.complete_graph(module_size)
    G = base_module.copy()

    # designate a hub (choose node 0 in each module) to connect between modules
    for level in range(2, levels + 1):
        # create 'branching-1' additional copies of current G and union them
        copies = [G.copy()]  # include the existing G
        for b in range(branching - 1):
            offset = sum(len(c) for c in copies)
            mapping = {n: n + offset for n in G.nodes()}
            H = nx.relabel_nodes(G, mapping)
            copies.append(H)
        # merge all copies into new_G and connect their hubs
        new_G = nx.Graph()
        for comp in copies:
            new_G = nx.union(new_G, comp)
        # connect hubs of each component (take the smallest node index in each comp as hub)
        comp_sizes = [len(c) for c in copies]
        start = 0
        hubs = []
        for sz in comp_sizes:
            hubs.append(start)   # the first node of each component is treated as hub
            start += sz
        # fully connect hubs (or you can connect them in a ring/tree — here we fully connect for cohesion)
        for i in range(len(hubs)):
            for j in range(i + 1, len(hubs)):
                new_G.add_edge(hubs[i], hubs[j])
        G = new_G

    return G

# create hierarchical with controlled size
G_hier = hierarchical_modular_network(levels=4, module_size=4, branching=2)
print(f"Hierarchical modular: nodes={G_hier.number_of_nodes()}, edges={G_hier.number_of_edges()}")

# ===== Simulate Attacks =====
networks = {'Configuration': G_config, 'Hierarchical': G_hier}
metrics = ['degree', 'clustering']

plt.figure(figsize=(12,5))
for i, metric in enumerate(metrics):
    plt.subplot(1,2,i+1)
    for name, G in networks.items():
        fr, giant = simulate_attack(G, metric)
        plt.plot(fr, giant, label=f"{name} (n={len(G)})")
    plt.xlabel("Fraction of nodes removed")
    if i == 0:
        plt.ylabel("Largest Component Fraction")
    plt.title(f"Attack by {metric}")
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()

# ===== Additional Observations =====
print("\nObservations / Additional Questions:")
print("1. Degree-based attack usually damages scale-free networks faster because hubs hold the network together.")
print("2. Clustering-based attack is often more effective in modular/hierarchical networks where clusters are tightly-knit.")
print("3. Protecting hubs (high-degree nodes) is most critical for scale-free network survival.")
print("4. Keeping topological info secret reduces risk of targeted attacks and improves network robustness.")
#----------------------------------------------------------------------------------#