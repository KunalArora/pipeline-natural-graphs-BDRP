import networkx as nx
import powerlaw
import os
import pydot
import matplotlib.pyplot as plt
import metis
from networkx.drawing.nx_pydot import write_dot
import numpy as np
import Queue
import random


colors = ['red','green','blue','yellow','orange','pink','black','gray','magenta']

def get_power_law_coefficient(graph):
    # obtain degree distribution
    degrees = []
    for node in graph.nodes_iter():
        degrees.append(len(graph.neighbors(node)))    
    # fit power law distribution
    dist = powerlaw.Fit(num_nodes)
    # obtain scaling factor
    return dist.power_law.alpha

def smooth_list(l):
    max_index = l.index(max(l))
    res = [x for x in l]
    for index in range(max_index)[::-1]:
        res[index] = random.randrange(int(0.8*res[index+1]), int(0.8*res[index+1]+1.3*res[index+2]))
    return res

def plot_power_law_degrees(graphs, graph_names, row_shape = 2, smooth=False, height=15):
    i = 0
    j = 0
    fig, axs = plt.subplots(row_shape,3, figsize=(20,height))
    for g, name in zip(graphs, graph_names):
      num_nodes = []
      for node in g.nodes_iter():
          num_nodes.append(len(g.neighbors(node)))

      x = range(1, max(num_nodes))
      y = [num_nodes.count(xx) for xx in x]
      if(smooth):
          y = smooth_list(y)      
      axs[i, j].plot(x, y)
      axs[i, j].set_xlabel("Degree", fontsize=12)
      axs[i, j].set_ylabel("Number of vertices", fontsize=12)
      axs[i, j].set_title(name + " graph", fontsize=14)
      axs[i, j].set_yscale('log')
      j+=1;
      if(j==3):
        i+=1
        j=0
    plt.show()

def load_graph_from_edge_file(filename, separator=" "):
    graph = nx.Graph()
    with open(filename, 'rb') as lines:
        for line in lines:
            s, t = line.strip().split(separator)
            graph.add_edge(s, t)
    return graph

def print_graph_stats(name, graph):
    num_nodes = []
    for node in graph.nodes_iter():
        num_nodes.append(len(graph.neighbors(node)))
        
    print(sorted(num_nodes, reverse=True))
    print(name, 
          str(graph.number_of_nodes())+" nodes",
          str(graph.number_of_edges())+" edges",
          "alpha="+str(get_power_law_coefficient(graph)))
    return sorted(num_nodes, reverse=True)

def print_graph_stats(name, graph):
    num_nodes = []
    for node in graph.nodes_iter():
        num_nodes.append(len(graph.neighbors(node)))
        
    print(sorted(num_nodes, reverse=True))
    print(name, 
          str(graph.number_of_nodes())+" nodes",
          str(graph.number_of_edges())+" edges",

          "alpha="+str(get_power_law_coefficient(graph)))


def generate_synthetic_graphs(size, scale, p_in, p_out, visualise=False, verbose=False):
    synthetic_graphs = []
    for seed in range(100):
        alpha = 3.0
        s = nx.utils.powerlaw_sequence(size, alpha)
        s = sorted(s, reverse=True)
        s[1] = s[2]+0.5*s[3]
        s[0] = s[1]+0.5*s[2]
        
        g = nx.random_partition_graph([int(scale*x) for x in s], p_in, p_out)
        if get_power_law_coefficient(g)<1.5 or get_power_law_coefficient(g)>5.0:
            continue
        # if g.number_of_edges()<g.number_of_nodes()*5:
            # continue
        if visualise:
            if float(g.number_of_edges())/g.number_of_nodes()>7:
                continue
            if verbose:
                pos = nx.spring_layout(g)
                nx.draw_networkx(g, pos, with_labels=False, node_size=25, width=0.1)
                plt.show()
            
        else:
            if verbose:
                plot_power_law_degrees(g)

        print(s)
        print(size, p_in, p_out, max(s), float(g.number_of_edges())/g.number_of_nodes(), get_power_law_coefficient(g))
        # print(sorted([int(x) for x in s], reverse=True))
        synthetic_graphs.append(g)
        # break
    return synthetic_graphs

synthetic_graphs = generate_synthetic_graphs(50, 5, 0.25, 0.01, visualise=True)[:9]

for graph in synthetic_graphs:
  nx.write_edgelist(graph, ("alpha"+str(get_power_law_coefficient(graph))))

# graph: power law vs something(edge cut)
def syn_graphs_1(synthetic_graphs, alphas):
    i=0
    j=0
    fig, axs = plt.subplots(3,3, figsize=(20,22))
    for graph, alpha in zip(synthetic_graphs[:9], alphas):
      (edgecuts, parts) = metis.part_graph(graph, 3)
      color_map = []
      for part, node in zip(parts, graph):
          color_map.append(colors[part])
      pos = nx.spring_layout(graph)
      nx.draw_networkx(graph, pos, node_color=color_map, with_labels=False, node_size=25, width=0.1, ax=axs[i, j])
      axs[i, j].set_title("alpha="+str(alpha))
      axs[i, j].axis('off') # tick_params(bottom=False,left=False)
      j+=1;
      if(j==3):
        i+=1
        j=0
    plt.show()

syn_graphs_1(synthetic_graphs_1, alphas)

# graph: power law vs something(edge cut)
def syn_graphs_2(synthetic_graphs):
    synthetic_edgecut_list = []
    synthetic_graph_edge_list = []
    synthetic_alphas = []
    for graph in synthetic_graphs:
      edgecut, no_of_edge = partition_graph(graph, 3)
      synthetic_edgecut_list.append(edgecut)
      synthetic_graph_edge_list.append(no_of_edge)
      synthetic_alphas.append(get_power_law_coefficient(graph))

    synthetic_results = [(x, y, z) for x, y, z in sorted(zip(synthetic_alphas, synthetic_edgecut_list, synthetic_graph_edge_list))]
    plt.plot([x for x,_, _ in synthetic_results], [x for _,x, _ in synthetic_results])
    plt.show()

syn_graphs_2(synthetic_graphs_2)

# power law distribution from synthetic
def syn_graphs_3(synthetic_graphs):
    for index, graph in enumerate(synthetic_graphs):
        if index%3==0:
            num_nodes = []
            for node in graph.nodes_iter():
                num_nodes.append(len(graph.neighbors(node)))
            x = range(1, max(num_nodes))
            y = [num_nodes.count(xx) for xx in x]
            plt.plot(x, y)
            plt.yscale('log')
            plt.xlabel("Degree")
            plt.ylabel("Number of vertices")
            plt.title("Synthetic graph with alpha = "+str(get_power_law_coefficient(graph)))
            plt.show()

ll = synthetic_graphs_3[:9]
plot_power_law_degrees(ll, ["alpha="+str(x) for x in alphas], smooth=True, row_shape=3, height=22)


# sampling region of real-world graph
twitter_graph = load_graph_from_edge_file("Twitter.txt")
facebook_page_graph = load_graph_from_edge_file("Facebook.txt")
github_graph = load_graph_from_edge_file("Github.txt")
slashdot_graph_2018 = load_graph_from_edge_file("Slashdot2018.txt")
slashdot_graph_2019 = load_graph_from_edge_file("Slashdot2019.txt")
epinions_graph = load_graph_from_edge_file("Epinions.txt")

graphs = [twitter_graph, facebook_page_graph, github_graph, slashdot_graph_2018, slashdot_graph_2019, epinions_graph]
graph_names = ["Twitter", "Facebook", "Github", "Slashdot-2018", "Slashdot-2019", "Epinions"]


def sample_graph(graph, graph_name, max_nodes=500):
    # get largest connected component
    graph = max(nx.connected_component_subgraphs(graph), key=len)

    # perform bfs
    source_node = list(graph.nodes_iter())[0]
    bfs_result = nx.bfs_successors(graph, source=source_node)
    subgraph_nodes = set()
    q = Queue.Queue()
    q.put(source_node)
    while not q.empty() and len(subgraph_nodes)<=max_nodes:
        current_node = q.get()
        subgraph_nodes.add(current_node)
        for neighbour in bfs_result.get(current_node, []):
            q.put(neighbour)

    # obtain induced subgraph
    sampled_graph = graph.subgraph(subgraph_nodes)
    return sampled_graph

samples = []
for graph_name, graph in zip(graph_names, graphs):
    samples.append(sample_graph(graph, graph_name))

def syn_graphs_100(graphs, names):
    i=0
    j=0
    fig, axs = plt.subplots(2,3, figsize=(18,12))
    for graph, name in zip(graphs, names):
      (edgecuts, parts) = metis.part_graph(graph, 3)
      color_map = []
      for part, node in zip(parts, graph):
          color_map.append(colors[part])
      pos = nx.spring_layout(graph)
      nx.draw_networkx(graph, pos, node_color=color_map, with_labels=False, node_size=25, width=0.1, ax=axs[i, j])
      axs[i, j].set_title("Sampled "+name+" graph, alpha="+str(get_power_law_coefficient(graph)))
      axs[i, j].axis('off')
      j+=1;
      if(j==3):
        i+=1
        j=0
    plt.show()

syn_graphs_100(samples, graph_names)

graphs = ll
graph_details = []
for graph in graphs:
    (edgecuts, parts) = metis.part_graph(graph, 3)
    graph_details.append((get_power_law_coefficient(graph), 
                          round(100.0*edgecuts/graph.number_of_edges(), 1),
                          graph.number_of_edges()))

graph_details = sorted(graph_details)
print(graph_details)

graph_names = ["alpha="+str(x) for x in alphas]
edgecut_list = [x for _, x, _ in graph_details]
graph_edge_list = [x for _, _, x in graph_details]

fig, ax1 = plt.subplots(1,1,figsize=(10,5))

color = 'tab:blue'
ax1.set_ylabel("Edgecuts (3 paritions)", fontsize=16, color=color)
ax1.set_xlabel('Graphs', fontsize=18)
ax1.plot(graph_names,edgecut_list, color=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('Number of edges in artificial graph', fontsize=18, color=color)  # we already handled the x-label with ax1
ax2.plot(graph_names, graph_edge_list, color=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()



twitter_graph = load_graph_from_edge_file("twitter_combined.txt")
facebook_page_graph = load_graph_from_edge_file("musae_facebook_edges.txt")
github_graph = load_graph_from_edge_file("musae_git_edges.txt")
slashdot_graph_2018 = load_graph_from_edge_file("soc-Slashdot0811.txt")
slashdot_graph_2019 = load_graph_from_edge_file("soc-Slashdot0902.txt")
epinions_graph = load_graph_from_edge_file("soc-Epinions1.txt")

graphs = [twitter_graph, facebook_page_graph, github_graph, slashdot_graph_2018, slashdot_graph_2019, epinions_graph]
graph_names = ["Twitter", "Facebook Page", "Github", "Slashdot-2018", "Slashdot-2019", "Epinions"]

alpha_list = []
max_degree_vertex = []
for name, graph in zip(graph_names, graphs):
    alpha_list.append(get_power_law_coefficient(graph))
    max_degree_vertex.append(print_graph_stats(name, graph)[0])


print(alpha_list)
print(max_degree_vertex)


graphs = [twitter_graph, facebook_page_graph, github_graph, slashdot_graph_2018, slashdot_graph_2019, epinions_graph]
edgecut_list = []
graph_edge_list = []
for graph in graphs:
  edgecut, no_of_edge = partition_graph(graph, 3)
  edgecut_list.append(edgecut)
  graph_edge_list.append(no_of_edge)

print(edgecut_list)
print(graph_edge_list)

print(alpha_list)
print(edgecut_list)

graph_names = ["Twitter", "Facebook", "Github", "Slashdot-2018", "Slashdot-2019", "Epinions"]

fig, ax1 = plt.subplots(1,1,figsize=(10,6))

color = 'tab:blue'
ax1.set_ylabel("Power-law degree distribution, alpha", fontsize=16)
ax1.set_xlabel('Graphs', fontsize=18)
ax1.bar(graph_names,alpha_list, color=color)
ax1.yaxis.grid(alpha=0.2)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('Maximum high degree vertex', fontsize=18)  # we already handled the x-label with ax1
ax2.plot(graph_names, max_degree_vertex, color=color)
ax2.tick_params(axis='y')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

graph_names = ["Twitter", "Facebook", "Github", "Slashdot-2018", "Slashdot-2019", "Epinions"]
real_alphas = [3.3, 3.22, 2.57, 3.53, 3.47, 3.64]
edgecut_list = [5.73, 4.75, 27.04, 34.27, 34.16, 24.8]
graph_edge_list = [1342310, 171002, 289003, 546487, 582533, 405740]

graph_names = [x+", "+str(y) for x, y in zip(graph_names, real_alphas)]

fig, ax1 = plt.subplots(1,1,figsize=(10,6))

color = 'tab:blue'
ax1.set_ylabel("Edgecuts (3 paritions)", fontsize=16, color=color)
ax1.set_xlabel('Graphs', fontsize=18)
ax1.plot(graph_names,edgecut_list, color=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('Number of edges in original graph', fontsize=18, color=color)  # we already handled the x-label with ax1
ax2.plot(graph_names, graph_edge_list, color=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

graphs = [twitter_graph, facebook_page_graph, github_graph, twitch_graph, slashdot_graph_2018, slashdot_graph_2019, epinions_graph]
paritions = [i for i in range(3,50)]
edgelist_dict_key = ["twitter", "facebook", "github", "twitch", "slashdot-2018", "slashdot-2019", "epinions"]
edgelist_dict = {"twitter": [], "facebook": [], "github": [], "twitch": [], "slashdot-2018": [], "slashdot-2019": [], "epinions": []}
i=-1
for graph in graphs:
  i+=1
  for partition in paritions:
    edgecut, no_of_edge = partition_graph(graph, partition)
    edgelist_dict[edgelist_dict_key[i]].append(edgecut)

print(edgelist_dict)


graph_names = ["Twitter", "Github", "Epinions", "Slashdot-2018", "Slashdot-2019", "Facebook", "Twitch"]

fig, ax1 = plt.subplots(1,1,figsize=(10,6))

color = 'tab:blue'
ax1.set_ylabel("Edgecuts", fontsize=16)
ax1.set_xlabel('Number of Partitions', fontsize=18)
ax1.set_xticks(np.arange(0,60,5))
ax1.plot(list(edgelist_dict["twitter"]), label="Twitter")
ax1.plot(list(edgelist_dict["github"]), label="Github")
ax1.plot(list(edgelist_dict["epinions"]), label="Epinions")
ax1.plot(list(edgelist_dict["slashdot-2018"]), label="Slashdot-2018")
ax1.plot(list(edgelist_dict["slashdot-2019"]), label="Slashdot-2019")
ax1.plot(list(edgelist_dict["facebook"]), label="Facebook")

ax1.legend()


fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

"""Plot POWER-LAW DEGREES"""

graphs = [twitter_graph, facebook_page_graph, github_graph, slashdot_graph_2018, 
          slashdot_graph_2019, epinions_graph]
graph_names = ["Twitter", "Facebook", "Github", "Slashdot-2018", "Slashdot-2019", "Epinions"]

plot_power_law_degrees(graphs, graph_names)

