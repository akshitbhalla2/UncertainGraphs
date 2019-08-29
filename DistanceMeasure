import sys as sys
import networkx as nx
import pandas as pd
import random as r
import math as m
import time as t
import pypref as p
import csv as csv
import matplotlib.pyplot as plt
from decimal import Decimal
from tqdm import tqdm
from functools import reduce

dataset_name = str(sys.argv[1])
distance_type = str(sys.argv[2])  ## ALL, MD, ED, ERD
num_query_nodes = int(sys.argv[3])  ## 2, 3, 5, 8, 10, 15, 20

global MyTime 
MyTime = []
samples = 100

# Graph Generation
print("Graph Generation")
start_time = t.time() # START TIME
G = nx.Graph()       
MyFile = "/home/akshit/Skyline_Uncertain_Graph/INTERNSHIP/datasets/" + dataset_name + ".csv"
MyData = pd.read_csv(MyFile)
edge_att = {}
for i in range(len(MyData)):
    a = int(MyData.iloc[i,0])
    b = int(MyData.iloc[i,1])
    c = Decimal(MyData.iloc[i,2])
    d = int(MyData.iloc[i,3])
    G.add_edge(a, b)
    edge_att[(a,b)] = {
        "p": c,
        "w": d
    }
nx.set_edge_attributes(G, edge_att)
MyTime.append(t.time() - start_time,) # END TIME
# nx.draw(G, with_labels = True)
# plt.show()

# Hop Tree (Upto 3 Neighbors) Generation
def CreateHopDic(MyGraph):
    HopDic = {}
    for i in tqdm(list(MyGraph.nodes)):
        HopDic[i] = {}
        temp_i = set(nx.neighbors(MyGraph, i)) | {i}
        for j in set(nx.neighbors(MyGraph, i)):
            val_j = set(nx.neighbors(MyGraph, j)) - temp_i
            HopDic[i][j] = val_j
            temp_j = temp_i | val_j
            if len(val_j) != 0:
                for k in val_j:
                    val_k = set(nx.neighbors(MyGraph, k)) - temp_j
                    HopDic[i][j] = HopDic[i][j] | val_k
    return HopDic
print("Hop Tree Generation")
start_time = t.time() # START TIME
HopDic_G = CreateHopDic(G)
MyTime.append(t.time() - start_time) # END TIME

def HopTreeNodes(num):
    neig = nx.neighbors(G, num)
    all_des = reduce(set.union, HopDic_G[num].values())
    return set(neig)|set(all_des)

# Query Vertex Generation
print("Query Vertex Generation")
start_time = t.time() # START TIME
Q = []
Q.append(r.randint(1,len(G.nodes)))
temp = set()
i = 0
while i < (num_query_nodes - 1):
    temp = temp|HopTreeNodes(Q[i])
    temp = temp - set(Q)
    i  = i + 1
    if len(temp) == 0:
        Q = []
        Q.append(r.randint(1,len(G.nodes)))
        i = 0
    else:
        Q.append(r.choice(list(temp)))
MyTime.append(t.time() - start_time) # END TIME
print(Q)
Q_names = []
for i in Q:
    Q_names.append("Q" + str(i))

# Pruning False Positives
def SSP_Prune(G, Q, HopDic_G):
    candidates = []
    MySet = set(G.nodes) - set(Q)
    for v in list(MySet):
        for n in list(HopDic_G[v].values()):
            if set(Q) <= n:
                break
        else:
            candidates.append(v)
    return candidates

# Candidate Generation
print("Candidate Generation")
start_time = t.time() # START TIME
CL = SSP_Prune(G, Q, HopDic_G)
MyTime.append(t.time() - start_time) # END TIME
CL_names = []
for i in CL:
    CL_names.append("D" + str(i))
print(len(CL))

# Create Sample Graph And Find Probability Of Existence
def Graph_and_Prob(G, edge_att):
    Gr = G.copy()
    prob = 1
    for i in range(len(G.edges)):
        test_prob = r.uniform(0,1)
        a, b = list(edge_att.keys())[i]
        if Gr[a][b]["p"] < test_prob:
            prob *= (1 - Gr[a][b]["p"])
            Gr.remove_edge(a,b)
        else:
            prob *= Gr[a][b]["p"]
    return Gr, prob

# Graph Sampling And Existence Probability
print("Graph Sampling And Existence Probability")
start_time = t.time() # START TIME
GraphProbList = []
for i in tqdm(range(samples)):
    Gr, prob = Graph_and_Prob(G, edge_att)
    GraphProbList.append((Gr, prob))
MyTime.append(t.time() - start_time) # END TIME

# Hop Tree (Upto 3 Neighbors) Generation
print("Hop Tree Generation")
start_time = t.time() # START TIME
HopDicList = []
for i in tqdm(range(samples)):
    HopDic = CreateHopDic(GraphProbList[i][0])
    HopDicList.append(HopDic)
MyTime.append(t.time() - start_time) # END TIME

# Listing Sum Of Weights Along Each Path
def FindDist(MyPath):
    MyDist = set()
    for i in range(len(MyPath)):
        w = 0
        for j in range(len(MyPath[i])-1):
            a = MyPath[i][j]
            b = MyPath[i][j+1]
            w += G[a][b]["w"]
        MyDist.add(w)
    return list(MyDist)

# Shortest Distance Query
def DisQ(Gr, HopDic, v, q, h):
    if h == 0:
        return None
    if q in list(nx.neighbors(Gr, v)):
        return Gr[v][q]["w"]
    else:
        flag = 0
        temp_distance = []
        for f in list(nx.neighbors(Gr, v)):
            if q in HopDic[v][f]:
                new_h = h - 1
                dist_f_q = DisQ(Gr, HopDic, f, q, new_h)
                try:
                    distance = Gr[v][f]["w"] + dist_f_q   
                    temp_distance.append(distance)
                    flag = 1
                except TypeError:
                    continue
        if flag == 0:
            return None
        else:
            MyDist = min(temp_distance)
            return MyDist

# Finding P(s,t)(d)
def PST(v, q, DistList, samples, GraphProbList, HopDicList):
    pstd = [0]*len(DistList)
    pstinf = 0
    for i in range(samples):
        Gr = GraphProbList[i][0]
        HopDic = HopDicList[i]
        short_dist = DisQ(Gr, HopDic, v, q, 3)
        if short_dist is not None:
            index = DistList.index(short_dist)
            pstd[index] += GraphProbList[i][1]
        else:
            pstinf += GraphProbList[i][1]
    return pstd, pstinf

# Major Distance Table 
def MD(G, CL, Q, CL_names, Q_names, samples, GraphProbList, HopDicList):
    print("Major Distance Table")
    MajorDistDF = pd.DataFrame(9999, index = CL_names, columns = Q_names) 
    for i, v in tqdm(enumerate(CL)):    
        for j, q in enumerate(Q):
            MyPath = list(nx.all_simple_paths(G, source = v, target = q, cutoff = 3))
            if len(MyPath) != 0:   
                DistList = FindDist(MyPath) 
                pstd, pstinf = PST(v, q, DistList, samples, GraphProbList, HopDicList) 
                index = pstd.index(max(pstd))
                MajorDistDF.iloc[i,j] = DistList[index]
    return MajorDistDF

# Expected Reliable Distance Table
def ERD(G, CL, Q, CL_names, Q_names, samples, GraphProbList, HopDicList):
    print("Expected Reliable Distance Table")
    ExpRelDistDF = pd.DataFrame(9999, index = CL_names, columns = Q_names) 
    for i, v in tqdm(enumerate(CL)):    
        for j, q in enumerate(Q):
            MyPath = list(nx.all_simple_paths(G, source = v, target = q, cutoff = 3))
            if len(MyPath) != 0:   
                DistList = FindDist(MyPath) 
                pstd, pstinf = PST(v, q, DistList, samples, GraphProbList, HopDicList) 
                numerator = zip(DistList, pstd) 
                ExpRelDistDF.iloc[i,j] = Decimal(0)
                for d, pstd in numerator:
                    ExpRelDistDF.iloc[i,j] += d*pstd            
                ExpRelDistDF.iloc[i,j] /= (1 - pstinf)
    return ExpRelDistDF

# Expected Distance Query
def ExpDist(G,v,q):
    MyList = []
    MyPath = list(nx.all_simple_paths(G, source = v, target = q, cutoff = 3))
    for i in range(len(MyPath)):
        p = 1
        w = 0
        for j in range(len(MyPath[i])-1):
            a = MyPath[i][j]
            b = MyPath[i][j+1]
            p *= G[a][b]["p"]
            w += G[a][b]["w"]
        MyList.append((p,w))
    dist = 0
    p_sum = 0
    for p, w in MyList:
        dist += p*w
        p_sum += p
    dist /= p_sum
    return dist

# Expected Distance Table 
def ED(G, CL, Q):
    print("Expected Distance Table")
    distances = {}
    for v in tqdm(CL):
        temp = []
        for q in Q:
            try:
                dist = ExpDist(G, v, q)
            except ZeroDivisionError:
                dist = 9999               
            temp.append(dist)
        distances[v] = temp
    ExpDistDF = pd.DataFrame(distances)
    ExpDistDF = ExpDistDF.transpose()
    ExpDistDF.columns = Q_names
    ExpDistDF.index = CL_names
    return ExpDistDF

# Finding Skylines
def FindSkyline(dataset):
    print("Finding Skylines")
    pref = p.low(dataset.columns[0])
    for i in list(dataset.columns)[1:]:
        pref *= p.low(i)
    return pref.psel(dataset)

# Writing To File
def WriteMyFile(MyFile, row):
    with open (MyFile, "a") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()

def CallMe(name, G, CL, Q, CL_names, Q_names, samples, GraphProbList, HopDicList):
    if name == "MD":
        start_time = t.time() # START TIME
        data = MD(G, CL, Q, CL_names, Q_names, samples, GraphProbList, HopDicList)
        MyTime.append(t.time() - start_time) # END TIME
    elif name == "ERD":
        start_time = t.time() # START TIME
        data = ERD(G, CL, Q, CL_names, Q_names, samples, GraphProbList, HopDicList)
        MyTime.append(t.time() - start_time) # END TIME
    elif name == "ED":
        start_time = t.time() # START TIME
        data = ED(G, CL, Q)
        MyTime.append(t.time() - start_time) # END TIME
    
    start_time = t.time() # START TIME
    sky = FindSkyline(data)
    MyTime.append(t.time() - start_time) # END TIME
    
    file1 = dataset_name + "_" + name + "_query_and_skyline_data.csv"
    file2 = dataset_name + "_" + name + "_query_candidate_skyline_size_data.csv"
    WriteMyFile('/home/akshit/Skyline_Uncertain_Graph/INTERNSHIP/results/' + dataset_name + '/' + file1, [len(Q), len(sky.index), Q, list(sky.index)])
    WriteMyFile('/home/akshit/Skyline_Uncertain_Graph/INTERNSHIP/results/' + dataset_name + '/' + file2, [len(Q), len(CL), len(sky.index)])

if distance_type == "MD":
    CallMe("MD", G, CL, Q, CL_names, Q_names, samples, GraphProbList, HopDicList)
elif distance_type == "ERD":
    CallMe("ERD", G, CL, Q, CL_names, Q_names, samples, GraphProbList, HopDicList)
elif distance_type == "ED":
    CalMe("ED", G, CL, Q, CL_names, Q_names, samples, GraphProbList, HopDicList)
elif distance_type == "ALL":
    CallMe("MD", G, CL, Q, CL_names, Q_names, samples, GraphProbList, HopDicList)
    CallMe("ERD", G, CL, Q, CL_names, Q_names, samples, GraphProbList, HopDicList)
    CallMe("ED", G, CL, Q, CL_names, Q_names, samples, GraphProbList, HopDicList)
else:
    print("Invalid Distance Measure...")

file3 = dataset_name + "_execution_times.csv"
WriteMyFile('/home/akshit/Skyline_Uncertain_Graph/INTERNSHIP/results/' + dataset_name + '/' + file3, MyTime)
