"""Module replicates the paper "A review of two decates of correlation, 
hierarchies, networks and clustering in financial markets using minimum spanning trees
Author: Rajan Subramanian
Date: 07/16/2020
"""

import numpy as np 
import yfinance as yf
import heapq
from itertools import combinations

class PriorityQueue:
    """Implementation of a priority queue using Heaps"""
    def __init__(self):
        self._queue = []
        self._index = 0 
    
    def push(self, item, priority):
        """pushes an item with a given priority
        Takes O(floor(logn)) time
        Args: 
        item:   the item we want to store
        priority:   default to min (use negative for max)
        """
        # if two items have the same priority, a secondary
        # index is added to avoid comparison failure
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1
    
    def pop(self):
        """removes the item with the highest priority
        Takes O(1) time
        """
        return heapq.heappop(self._queue)
    
    def __len__(self):
        return len(self._queue)
    
    def is_empty(self):
        return len(self) == 0

class Graph:
    """implementation of a graph using adjacency map
    For a pair of vertices (u,v), (u,z) that has an edges E, F
    is represented by {u: {v: E, z: F}}"""

    """nested Vertex and Edge classes"""
    class __Vertex: 
        """vertex structure for graph"""
        __slots__ = '_value'

        def __init__(self, val):
            self._value = val
        
        def get_value(self):
            """return value associated with this vertex"""
            return self._value 
        
        def __hash__(self):
            """allows vertex to be a key in a dictionary"""
            return hash(id(self))
        
        def __repr__(self):
            return """Vertex({!r})""".format(self._value)
    
    # --  nested edge class
    class __Edge:
        """Implements the edge structure that returns edge associated
            with vertex (u,v)
        """
        # light weight edge structure
        __slots__ = '_start', '_end', '_value'

        def __init__(self, u, v, val):
            self._start = u 
            self._end = v 
            self._value = val 
        
        def __repr__(self):
            insert = (self._start, self._end, self._value)
            return """Edge(({!r}, {!r}): {:.2f}""".format(*insert)
        
        def endpoint(self):
            """return (u,v) as a tuple for vertices u and v"""
            return (self._start, self._end)
        
        def opposite(self, u):
            """return vertex opposite of u on this edge"""
            return self._end if u is self._start else self._start
        
        def get_value(self):
            """return value associated with this edge"""
            return self._value 
        
        def get_items(self):
            """returns edge attributes as a tuple
            Helpful for visualizing nodes and their edge weights"""
            return (self._start._value, self._end._value, self._value)
        
        def __hash__(self):
            return hash((self._start, self._end))
    
    # -- beginning of graph definition
    def __init__(self, directed=False):
        """create an empty graph undirected by default
        Graph is directed if parameter is set to True
        """
        self._out = {}
        self._in = {} if directed else self._out
    
    def is_directed(self):
        """return True if graph is directed, False otherwise"""
        return self._in is not self._out
    
    def count_vertices(self):
        """returns the count of total vertices in a graph"""
        return len(self._out)
    
    def get_vertices(self):
        """returns iteration of vertices keys"""
        return self._out.keys()
    
    def count_edges(self):
        """return count of total edges in a graph"""
        total_edges = sum(len(self._out[v]) for v in self.get_vertices())
        return total_edges if self.is_directed() else total_edges // 2
    
    def get_edge(self, u, v):
        """returns the edge between u and v, None if its non existent"""
        return self._out[u].get(v)
    
    def get_edges(self):
        """returns iteration of all unique edges in a graph"""
        seen = set()
        for inner_map in self._out.values():
            seen.update(inner_map.values())
        return seen 
    
    def degree(self, u, outgoing=True):
        """return total outgoing edges incident to vertex u in the graph
        for directed graph, optional parameter counts incoming edges
        """
        temp = self._out if outgoing else self._in 
        return len(temp[u])
    
    def iter_incident_edges(self, u, outgoing=True):
        """returns iteration of all outgoing edges incident to vertex u in this graph
        for directed graph, optional paramter will receive incoming edges
        """
        temp = self._out if outgoing else self._in 
        for edge in temp[u].values():
            yield edge 
    
    def insert_vertex(self, val=None):
        """insert and return a new vertex with value val"""
        u = self.__Vertex(val)
        self._out[u] = {}
        if self.is_directed():
            self._in[u] = {}
        return u 
    
    def insert_edge(self, u, v, val=None):
        """insert and return a new edge from u to v with  value val (identifies the edge)"""
        edge = self.__Edge(u,v,val)
        self._out[u][v] = edge 
        self._in[v][u] = edge

class Cluster:
    """Union find structure for use in Kruskal's algo
        to check if undirected graph contains cycle or not
    """
    # - Nested Position class to keep track of parent
    class Position:
        """Light weight structure for slots"""
        __slots__ = '_container', '_value', '_size', '_parent'
        def __init__(self, container, value):
            self._container = container 
            self._value = value
            self._size = 1 
            self._parent = self 
        
        def get_value(self):
            return self._value
        
    def make_cluster(self, value):
        return self.Position(self, value)
    
    def find(self, p):
        """find group contianing p and return position of its leader"""
        if p._parent != p:
            p._parent = self.find(p._parent)
        return p._parent
    
    def union(self, p, q):
        """merges group containing element p and q (if distinct)"""
        a = self.find(p)
        b = self.find(q)
        if a is not b:
            if a._size > b._size:
                b._parent = a 
                a._size += b._size 
            else:
                a._parent = b 
                b._size += a._size

class MinimumSpanningTrees:
    """Implementation of minimum spanning tree using Kruskal's algorithm
    Args: 
    None

    Attributes:
    distance:   pandas DataFrame

    Returns: 
    Minimum Spanning Tree using Kruskal's aglo
    """
    # -- Nested Price Class
    class Price:
        """Get the prices from yahoo finance and calculates their distances
        Args:
        start: start date in format YYYY-MM-DD in string
        end:   end date in format YYYY-MM-DD in string

        Returns: 
        distances calculated from correlation matrix (pandas object)

        Notes: 
        prices are downloaded from yahoo finance using yfinance module
        log returns are calculated using the formula 
            rt = log(1 + Rt) where Rt is the percentage return
        Distances are calculated using the formula: 
            dij = sqrt(2 * (1 - pij))
        """
        def __init__(self, start, end):
            self.start = start 
            self.end = end 
        
        def get_prices(self, col='Adj Close'):
            """gets the adjusted close price from yahoo finance
            Args: 
            col: string supported types are High, low, Adj Close
            """
            # - set ticker names here 
            ticker_names = """\
            TSLA SPY MSFT MMM ABBV ABMD ACN ATVI ADBE AMD 
            AES AFL APD AKAM ALB ARE ALLE GOOG AAL AMT
            ABC AME AMGN APH ADI AMAT APTV BKR BAX BDX
            BIO BA BKNG BWA BXP BSX BMY COG COF KMX
            CBOE CE CNC CF DLTR EFX FIS GD GE GS
            """
            return yf.download(ticker_names, start=self.start, end=self.end, progress=False)[col]
        
        def _calculate_correlation(self, prices):
            """calculates the correlation given prices
            Args: 
            prices: pandas dataframe

            Returns: 
            correlation:  pandas dataframe
            """
            return np.log(1 + prices.pct_change()).corr()
            
        def get_distance(self):
            """Computes the distance given correlation
                dij := sqrt( 2(1 - pij) )
            """
            prices = self.get_prices()
            pairwise = self._calculate_correlation(prices)
            distance = np.sqrt(2 * (1 - pairwise))
            return distance

    # ---MST Kruskals algorithm
    def __init__(self, start, end):
        self.distance = self.Price(start, end).get_distance()
    
    def create_graph(self):
        """creates a graph with vertices and edges from distance
        Args:
        None
        
        Returns:
        graph object
        """
        g = Graph()
        share_names = iter(list(self.distance))
        vertices = iter([g.insert_vertex(v) for v in share_names])
        edges = []
        for c in combinations(vertices, 2):
            # get the vertices
            u, v = c
            # get the distance weight
            w = self.distance.loc[u.get_value(), v.get_value()]
            # create edge
            g.insert_edge(u, v, w)
        return g

    def mst_kruskal(self, g):
        """compute minimum spanning tree using kruskal's algorithm
        Args:
        g:     Graph with a adjacy map structure
        
        Returns:
        list of graph's edges where edges are weights
        """
        tree = []  # stores edges of a spanning tree
        pq = PriorityQueue()  # to store the minimum edges of a graph
        cluster = Cluster()
        position = {}  # map each node to partition array

        for v in g.get_vertices():
            position[v] = cluster.make_cluster(v)
        
        for e in g.get_edges():
            pq.push(e, e.get_value())
        
        size = g.count_vertices()
        while len(tree) != size - 1 and not pq.is_empty():
            weight, _, edge = pq.pop()
            u, v = edge.endpoint()
            a = cluster.find(position[u])
            b = cluster.find(position[v])
            if a != b: 
                tree.append(edge)
                cluster.union(a, b)
        return tree
    
    def draw_graph(self, mst_tree):
        """Plots the minimum spanning tree
        Args: 
        mst_tree:  list of tree objects with minimum edges from Edge Class
        
        Returns: 
        None:       plot object containing the MST
        """
        import networkx as nx
        import matplotlib.pyplot as plt 
        g = nx.Graph()
        # get the edge vertices and edge weights from mst_tree
        items = (e.get_items() for e in mst_tree)
        # add the edges in the networkx graph for plotting
        g.add_weighted_edges_from(items)
        nx.draw(g, with_labels=True);
        plt.draw();




        






        