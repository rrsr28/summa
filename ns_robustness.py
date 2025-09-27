import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from powerlaw import Fit
import random

# ===================== Utility Functions =====================
def compute_metrics_and_distribution(G, label="Network"):
    """Compute network metrics, degree distribution, and power-law fit."""
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
    """Remove fraction of nodes randomly or targeted (by degree)."""
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
    """Simulate percolation and return largest component fraction at each step."""
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
    """Estimate percolation threshold where largest component <= threshold."""
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
    """Generate ER, BA, and WS networks."""
    ER = nx.erdos_renyi_graph(n, p_er)
    BA = nx.barabasi_albert_graph(n, m_ba)
    WS = nx.watts_strogatz_graph(n, k_ws, p_ws)
    return ER, BA, WS

# ===================== Main Analysis =====================
if __name__ == "__main__":
    print("=== Complex Network Robustness and Scale-Free Analysis ===")
    n = 1000
    m_ba = 3
    p_er = 2*m_ba/(n-1)  # ER probability to match BA average degree ~6
    fraction_remove = 0.2  # 20% nodes removed

    # Generate networks
    ER, BA, WS = generate_networks(n=n, p_er=p_er, m_ba=m_ba)
    print("Networks generated: ER, BA, WS")

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
