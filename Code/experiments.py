import numpy as np
import networkx as nx
import json

from psi_score import psi_score_old, power_psi, psi_solve
from pagerank import pagerank
from utils import create_graph_from_file, stats_init


def experiment_routine(L, lambdas, mus, epsilons, stats, alpha=0.85, old=True, n_max=1000):
    if stats["name"] == "dblp":
        psi_true, _ = psi_solve(L, lambdas, mus)
    
    for eps in epsilons:
        if old:
            psi_pow, d_pow, n_msg_pow, n_mult_pow = psi_score_old(L, lambdas, mus, eps=eps, n_max=n_max)
            stats["power_nf"]["n_mult"].append(n_mult_pow)
            stats["power_nf"]["times"].append(d_pow)
            stats["power_nf"]["n_msg"].append(n_msg_pow)


        psi_new, d_new, n_msg_new, n_mult_new = power_psi(L, lambdas, mus, eps=eps, n_max=n_max)
        stats["power_psi"]["n_mult"].append(n_mult_new)
        stats["power_psi"]["times"].append(d_new)
        stats["power_psi"]["n_msg"].append(n_msg_new)


        pr, d_pr, n_msg_pr, n_mult_pr = pagerank(A, alpha=alpha, eps=eps, max_iter=n_max)
        stats["pagerank"]["n_mult"].append(n_mult_pr)
        stats["pagerank"]["times"].append(d_pr)
        stats["pagerank"]["n_msg"].append(n_msg_pr)

        if stats["name"] == "dblp":
            err_pow = np.linalg.norm(psi_pow - psi_true, ord=2)/np.linalg.norm(psi_true, ord=2)
            stats["power_nf"]["errors"].append(err_pow)

            err_new = np.linalg.norm(psi_new - psi_true, ord=2)/np.linalg.norm(psi_true, ord=2)
            stats["power_psi"]["errors"].append(err_new)
            
            err_pr = np.linalg.norm(pr - psi_true, ord=2)/np.linalg.norm(psi_true, ord=2)
            stats["pagerank"]["errors"].append(err_pr)
    return stats

dblp_path = "../Data/cit-DBLP.edges"  
twitter_path = "../Data/munmun_twitter_social/out.munmun_twitter_social"
fb_path = "../Data/facebook-wosn-links/out.facebook-wosn-links"
arxiv_path = "../Data/cit-HepPh/out.cit-HepPh"

Epsilons = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1]

### DBLP psi-score
G = create_graph_from_file(dblp_path)
A = nx.to_scipy_sparse_matrix(G)
L = nx.to_dict_of_lists(G)
N = G.number_of_nodes()

np.random.seed(10)
lam = np.random.random(N)
mus = np.random.random(N)

dblp = stats_init("dblp")
dblp = experiment_routine(L, lam, mus, Epsilons, dblp, alpha=0.85)

with open("../Code/output/paper/dblp/dblp_psi.json", "w") as fp:
    json.dump(dblp, fp)

### DBLP PageRank

dblp_pagerank = stats_init("dblp")

lam = N * [0.15]
mus = N * [0.85]

dblp_pagerank = experiment_routine(L, lam, mus, Epsilons, dblp_pagerank, alpha=0.85)

with open("../Code/output/paper/dblp/dblp_pagerank.json", "w") as fp:
    json.dump(dblp_pagerank, fp)

del dblp_pagerank
del dblp

### Twitter psi-score

G = create_graph_from_file(twitter_path)
A = nx.to_scipy_sparse_matrix(G)
L = nx.to_dict_of_lists(G)
N = G.number_of_nodes()

np.random.seed(10)
lam = np.random.random(N)
mus = np.random.random(N)

twitter = stats_init("twitter")
twitter = experiment_routine(L, lam, mus, Epsilons, twitter, alpha=0.85)

with open("../Code/output/paper/twitter/twitter_psi.json", "w") as fp:
    json.dump(twitter, fp)

### Twitter PageRank
lam = N * [0.15]
mus = N * [0.85]

twitter_pagerank = stats_init("twitter")
twitter_pagerank = experiment_routine(L, lam, mus, Epsilons, twitter_pagerank, alpha=0.85)

with open("../Code/output/paper/twitter/twitter_pagerank.json", "w") as fp:
    json.dump(twitter_pagerank, fp)

del twitter_pagerank
del twitter

### hep_ph psi-score

G = create_graph_from_file(arxiv_path, sep="\t")
A = nx.to_scipy_sparse_matrix(G)
L = nx.to_dict_of_lists(G)
N = G.number_of_nodes()

np.random.seed(10)
lam = np.random.random(N)
mus = np.random.random(N)

hep_ph = stats_init("hep_ph")
hep_ph = experiment_routine(L, lam, mus, Epsilons, hep_ph, alpha=0.85)

with open("../Code/output/paper/hep_ph/hep_ph_psi.json", "w") as fp:
    json.dump(hep_ph, fp)

### hep_ph PageRank
lam = N * [0.15]
mus = N * [0.85]

hep_ph_pagerank = stats_init("hep_ph")
hep_ph_pagerank = experiment_routine(L, lam, mus, Epsilons, hep_ph_pagerank, alpha=0.85)

with open("../Code/output/paper/hep_ph/hep_ph_pagerank.json", "w") as fp:
    json.dump(hep_ph_pagerank, fp)

del hep_ph_pagerank
del hep_ph

### Facebook psi-score

G = create_graph_from_file(fb_path)
A = nx.to_scipy_sparse_matrix(G)
L = nx.to_dict_of_lists(G)
N = G.number_of_nodes()

np.random.seed(10)
lam = np.random.random(N)
mus = np.random.random(N)

facebook = stats_init("facebook")
facebook = experiment_routine(L, lam, mus, Epsilons, facebook, alpha=0.85)

with open("../Code/output/paper/facebook/facebook_psi.json", "w") as fp:
    json.dump(facebook, fp)

### Facebook PageRank
lam = N * [0.15]
mus = N * [0.85]

facebook_pagerank = stats_init("facebook")
facebook_pagerank = experiment_routine(L, lam, mus, Epsilons, facebook_pagerank, alpha=0.85)

with open("../Code/output/paper/facebook/facebook_pagerank.json", "w") as fp:
    json.dump(facebook_pagerank, fp)

del facebook_pagerank
del facebook

