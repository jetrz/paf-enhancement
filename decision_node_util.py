import torch
from torch_geometric.utils import to_networkx
import networkx as nx

def add_decision_nodes(g, aux, params):
    print("Adding decision nodes...")

    nx_graph = to_networkx(data=g, node_attrs=aux['node_attrs'], edge_attrs=aux['edge_attrs'])

    nx_graph, params = create_gt(nx_graph, params)

    nx_graph = add_soft_binary_gt_7classes(nx_graph)
    nx_graph = add_binary_gt(nx_graph)
    nx_graph = add_decision_attr(nx_graph)

    new_node_attrs, new_edge_attrs = ['decision_node'], ['gt_17c', 'gt_soft', 'gt_bin', 'decision_edge_gt_only_pos', 'decision_edge_gt_only_neg']
    g.decision_node = torch.tensor([nx_graph.nodes[node.item()]['decision_node'] for node in g.N_ID])
    edges = list(zip(g.edge_index[0].tolist(), g.edge_index[1].tolist()))
    for attr in new_edge_attrs:
        curr_dict = { edge : nx_graph.edges[edge][attr] for edge in nx_graph.edges }
        g[attr] = torch.tensor([curr_dict[edge] for edge in edges])

    aux['node_attrs'].extend(new_node_attrs)
    aux['edge_attrs'].extend(new_edge_attrs)

    return g, aux

def create_gt(graph, params):
    read_start_dict = nx.get_node_attributes(graph, "read_start")
    read_end_dict = nx.get_node_attributes(graph, "read_end")
    read_strand_dict = nx.get_node_attributes(graph, "read_strand")
    read_chr = nx.get_node_attributes(graph, 'read_chr')
    graph_edges_all, cross_chr_edges = check_for_cross_chromosome(graph, read_chr)
    all_chrs = list(set(read_chr.values()))
    gt_dict = {}

    for edge in cross_chr_edges:
        gt_dict[edge] = 0

    for chr in all_chrs:
        #  get all graph edges of chromosome
        graph_edges = set()
        for edge in graph_edges_all:
            src, dst = edge
            if read_chr[src] == chr:
                graph_edges.add(edge)
                
        # Split graph in + and - strand
        pos_edges, neg_edges, cross_strand_edges = split_strands(graph_edges, read_strand_dict)
        # Get correct edges
        pos_correct, pos_skip_sequence = get_correct_edges(pos_edges, read_start_dict, read_end_dict, positive=True)
        neg_correct, neg_skip_sequence = get_correct_edges(neg_edges, read_start_dict, read_end_dict, positive=False)

        pos_graph = create_subgraph_dip(pos_correct)
        neg_graph = create_subgraph_dip(neg_correct)
        optimal_edges_pos, deadend_edges_pos, params = find_deadends(pos_graph, params, read_start_dict, read_end_dict, positive=True)
        optimal_edges_neg, deadend_edges_neg, params = find_deadends(neg_graph, params, read_start_dict, read_end_dict, positive=False)
        deadend_edges = deadend_edges_pos | deadend_edges_neg
        optimal_edges = optimal_edges_pos | optimal_edges_neg

        print(f"chr edges: {len(graph_edges)}")
        print(f"pos edges: {len(pos_edges)} neg edges: {len(neg_edges)} cross strand edges: {len(cross_strand_edges)} sum: {len(pos_edges) + len(neg_edges) + len(cross_strand_edges)}" )

        for edge in cross_strand_edges:
            gt_dict[edge] = 1
        for edge in pos_skip_sequence | neg_skip_sequence:
            gt_dict[edge] = 3
        for edge in deadend_edges:
            gt_dict[edge] = 4
        for edge in optimal_edges:
            gt_dict[edge] = 5

    if len(gt_dict) != len(graph.edges()):
        print("ERROR: Not all edges have been labeled!")
        print(len(cross_chr_edges) , len(graph_edges_all), len(cross_chr_edges) + len(graph_edges_all))

        print(f"Edges: {len(graph_edges_all)}, Labeled: {len(gt_dict)}")
        exit()
    nx.set_edge_attributes(graph, gt_dict, "gt_17c")

    """# Dictionary to store the occurrence of each value
    value_counts = {}

    # Loop through each value in the original dictionary
    for value in gt_dict.values():
        # If the value is already in the value_counts dict, increment its count
        if value in value_counts:
            value_counts[value] += 1
        # Otherwise, add the value to the value_counts dict with a count of 1
        else:
            value_counts[value] = 1

    # Print the counts of each value
    for value, count in value_counts.items():
        print(f"Value: {value}, Occurrences: {count}")
    exit()"""

    return graph, params

def split_strands(graph_edges, read_strand_dict):
    pos_edges, neg_edges, cross_strand_edges = set(), set(), set()
    for edge in graph_edges:
        src, dst = edge
        if read_strand_dict[src] == -1 and read_strand_dict[dst] == -1:
            neg_edges.add(edge)
        elif read_strand_dict[src] == 1 and read_strand_dict[dst] == 1:
            pos_edges.add(edge)
        else:
            cross_strand_edges.add(edge)

    return pos_edges, neg_edges, cross_strand_edges

def check_for_cross_chromosome(graph, read_chr):
    graph_edges_all = graph.edges()
    cross_chr_edges = set()
    other_edges = set()

    for edge in graph_edges_all:
        src, dst = edge
        if read_chr[src] != read_chr[dst]:
            cross_chr_edges.add(edge)
        else:
            other_edges.add(edge)
    return other_edges, cross_chr_edges

def get_correct_edges(edges, read_start_dict, read_end_dict, positive=True):
    # only real connections of true overlaps
    skip_sequence, correct = set(), set()

    for edge in edges:
        src, dst = edge
        read_start_src = read_start_dict[src]
        read_end_src = read_end_dict[src]
        read_start_dst = read_start_dict[dst]
        read_end_dst = read_end_dict[dst]

        # if not Cross-Haplotype, check if it's a skip-sequence edge
        is_correct = is_correct_edge(read_start_src, read_end_src,
                                                read_start_dst, read_end_dst, positive)
        if is_correct:
            correct.add(edge)
        else:
            skip_sequence.add(edge)

    return correct, skip_sequence

def is_correct_edge(src_start, src_end, dst_start, dst_end, positive):
    # contained:
    if (src_start <= dst_start and src_end >= dst_end) or (src_start >= dst_start and src_end <= dst_end):
        return True  # Contained Edges are good Edges for HiFiasm Graphs
    # overlap
    if positive:
        #read_start_dict[dst] < read_end_dict[src] and read_start_dict[dst] > read_start_dict[src]
        return dst_start < src_end and dst_start > src_start
        #return src_start < dst_end and src_start > dst_start
    else:
        #neg_edge_condition = (read_start_dict[src] < read_end_dict[dst] and read_start_dict[src] > read_start_dict[dst] and read_strand_dict[src] == -1 and read_strand_dict[dst] == -1 and read_variant_dict[dst] == read_variant_dict[src])
        return src_start < dst_end and src_start > dst_start
        #return src_end > dst_start and src_end < dst_end

def liftover_convert_coordinate(liftover_object, chr, position):
    new_pos = liftover_object.convert_coordinate(chr, position)
    if new_pos:
        return new_pos[0][1]
    else:
        return position

def create_subgraph_dip(edges):
    active_edges = set()
    for e in edges:
        active_edges.add(e)

    sub_graph = nx.DiGraph()
    sub_graph.add_edges_from(active_edges)
    return sub_graph

def find_deadends(graph, params, read_start_dict, read_end_dict, positive=False):
    # components = [] # not for gt (later used)
    all_nodes = graph.nodes()
    graph_edges = set(graph.edges())
    gt_edges = set()
    if positive:
        final_node = max(all_nodes, key=lambda x: read_end_dict[x])
        highest_node_reached = min(all_nodes, key=lambda x: read_end_dict[x])
    else:
        final_node = min(all_nodes, key=lambda x: read_start_dict[x])
        highest_node_reached = max(all_nodes, key=lambda x: read_start_dict[x])

    while all_nodes:
        if positive:
            start_node = min(all_nodes, key=lambda x: read_start_dict[x])
        else:
            start_node = max(all_nodes, key=lambda x: read_end_dict[x])

        # try finding a path and report the highest found node during the dfs
        current_graph = graph.subgraph(all_nodes)
        full_component = set(nx.dfs_postorder_nodes(current_graph, source=start_node))

        if positive:
            highest_node_in_component = max(full_component, key=lambda x: read_end_dict[x])
        else:
            highest_node_in_component = min(full_component, key=lambda x: read_start_dict[x])

        current_graph = graph.subgraph(full_component)
        component = set(nx.dfs_postorder_nodes(current_graph.reverse(copy=True), source=highest_node_in_component))
        current_graph = graph.subgraph(component)

        # if the path doesnt go further then an already existing chunk - dont add any edges to gt
        not_reached_highest = (positive and (
                read_end_dict[highest_node_in_component] < read_end_dict[highest_node_reached])) \
                                or (not positive and (
                read_start_dict[highest_node_in_component] > read_start_dict[highest_node_reached]))
        if len(component) <= 2 or not_reached_highest:
            all_nodes = all_nodes - full_component
            for n in full_component:
                if positive:
                    params['deadends'][n] = read_end_dict[highest_node_in_component] - read_end_dict[highest_node_reached]
                else:
                    params['deadends'][n] = read_start_dict[highest_node_reached] - read_start_dict[highest_node_in_component]
            continue
        else:
            highest_node_reached = highest_node_in_component
        gt_edges = set(current_graph.edges()) | gt_edges
        # print("finish component")
        if highest_node_reached == final_node:
            break
        all_nodes = all_nodes - full_component

    return gt_edges, graph_edges - gt_edges, params

def add_soft_binary_gt_7classes(nx_graph):
    gt_17c = nx.get_edge_attributes(nx_graph, 'gt_17c')

    gt = {}
    for e in gt_17c.keys():
        if gt_17c[e] in {0, 1}: # Strand Change
            gt[e] = 0
        elif gt_17c[e] ==3: # Skip Sequence
            gt[e] = 0.1
        elif gt_17c[e] == 5:
            gt[e] = 1
        elif gt_17c[e] == 4:
            gt[e] = 0.2

    nx.set_edge_attributes(nx_graph, gt, 'gt_soft')

    return nx_graph

def add_decision_attr(nx_graph):
    crucial_dec_nodes, only_pos_decision_edges, only_neg_decision_edges = get_binary_crucial_decisions(nx_graph, 'gt_soft')

    nx.set_node_attributes(nx_graph, crucial_dec_nodes, 'decision_node')
    nx.set_edge_attributes(nx_graph, only_pos_decision_edges, 'decision_edge_gt_only_pos')
    nx.set_edge_attributes(nx_graph, only_neg_decision_edges, 'decision_edge_gt_only_neg')

    return nx_graph

def get_binary_crucial_decisions(nx_graph, dec_attr):
    #gt_m_soft
    gt = nx.get_edge_attributes(nx_graph, dec_attr)

    crucial_decision_nodes = {}
    only_pos_decision_edges = {}
    only_neg_decision_edges = {}


    for e in nx_graph.edges():
        only_neg_decision_edges[e] = 0
        only_pos_decision_edges[e] = 0

    crucial_decisions_count = 0

    for source in nx_graph.nodes():
        crucial_decision_nodes[source] = 0
        best_edge_type = None
        good_out_edges = []
        for target in nx_graph.successors(source):
            edge = (source, target)
            if best_edge_type == None:
                best_edge_type = gt[edge]
                good_out_edges.append(edge)
            elif gt[edge] > best_edge_type:
                crucial_decision_nodes[source] = 1
                best_edge_type = gt[edge]
                good_out_edges = [edge]
            elif gt[edge] == best_edge_type:
                good_out_edges.append(edge)
            else: #  gt[edge] < best_edge_type
                crucial_decision_nodes[source] = 1

        if crucial_decision_nodes[source] == 1:
            crucial_decisions_count +=1
            for target in nx_graph.successors(source):
                only_neg_decision_edges[(source, target)] = 1
            for edge in good_out_edges:
                only_pos_decision_edges[edge] = 1
                only_neg_decision_edges[edge] = 0

            #print("good out edges vs total edges:", len(good_out_edges), len(list(nx_graph.successors(source))))
            if (len(good_out_edges)==0 or len(good_out_edges) == len(list(nx_graph.successors(source)))):
                print("problem with good edges")

    print(f"Chr has {crucial_decisions_count} crucial decisions out of {nx_graph.number_of_nodes()} nodes.")
    return crucial_decision_nodes, only_pos_decision_edges, only_neg_decision_edges

def add_binary_gt(nx_graph):
    gt_17c = nx.get_edge_attributes(nx_graph, 'gt_17c')
    gt_bin = {}
    for e in gt_17c.keys():
        if abs(gt_17c[e]) == 5 or gt_17c[e] == 10:
            gt_bin[e] = 1
        else:
            gt_bin[e] = 0

    nx.set_edge_attributes(nx_graph, gt_bin, 'gt_bin')

    return nx_graph