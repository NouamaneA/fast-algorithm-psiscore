import scipy.sparse as sps
import networkx as nx

def dict_to_sparse_matrix(d, shape, adj=False):
    N = len(d)
    row = []
    col = []
    data = []
    for i in d:
        row += len(d[i])*[i]
        col += list(d[i])
        if adj:
            data += len(d[i])*[1]
        else:
            data += list(d[i].values())
    return sps.csr_matrix((data, (row, col)), shape=shape)

def l_plus_m(L, ls, ms):
    lpms = dict()
    for i in L.keys():
        lpms[i] = sum([ ls[l] + ms[l] for l in L[i] ])
    return lpms

def leaders(i, F):
    L_i = []
    N = len(F)
    for j in range(N):
        if F[i, j] > 0:
            L_i.append(j)
    return L_i

def followers(i, F):
    F_i = []
    N = len(F)
    for j in range(N):
        if F[j, i] > 0:
            F_i.append(j)
    return F_i

def create_graph_from_file(file_path, directed=True, sep=" "):
    edges_list = []
    with open(file_path, 'r') as f:
        for l in f.readlines():
            l = l.split(sep=sep)
            if l[0].isdigit():
                u, v = int(l[0]), int(l[1])
                edges_list.append((u-1, v-1))
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G.add_edges_from(edges_list)
    return G

def stats_init(name):
    stats = {
        "name": name,
        "power_nf": {
            "n_msg": [],
            "n_mult": [],
            "times": [],
            "errors": []
        },
        "power_psi": {
            "n_msg": [],
            "n_mult": [],
            "times": [],
            "errors": []
        },
        "pagerank": {
            "n_msg": [],
            "n_mult": [],
            "times": [],
            "errors": []
        }
    }
    return stats