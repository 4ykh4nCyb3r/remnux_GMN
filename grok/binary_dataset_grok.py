import json
import numpy as np
import collections
import random

GraphData = collections.namedtuple('GraphData', [
    'from_idx', 'to_idx', 'node_features', 'edge_features', 'graph_idx', 'n_graphs'])

class BinaryDataset:
    def __init__(self, json_file):
        print(f"[*] Loading dataset from {json_file}")
        with open(json_file, 'r') as f:
            functions = json.load(f)
        
        self.graphs = self._parse_functions(functions)
        self.function_names = list(self.graphs.keys())
        print(f"[*] Loaded {len(self.function_names)} functions")

    def _parse_functions(self, functions):
        parsed = {}
        for func in functions:
            name = func['name']
            node_features = np.array(func['node_features'], dtype=np.float32)
            edges = func.get('edges', [])
            
            from_idx = [e[0] for e in edges]
            to_idx = [e[1] for e in edges]
            
            edge_features = np.ones((len(from_idx), 1), dtype=np.float32) if from_idx else np.zeros((0, 1), dtype=np.float32)
            
            parsed[name] = {
                'node_features': node_features,
                'edge_features': edge_features,
                'from_idx': np.array(from_idx, dtype=np.int32),
                'to_idx': np.array(to_idx, dtype=np.int32)
            }
        return parsed

    def _pack_batch(self, graphs_list):
        # Same as before (unchanged)
        from_idx_list, to_idx_list, node_features_list, edge_features_list, graph_idx_list = [], [], [], [], []
        node_offset = 0
        for i, g in enumerate(graphs_list):
            n_nodes = g['node_features'].shape[0]
            node_features_list.append(g['node_features'])
            edge_features_list.append(g['edge_features'])
            from_idx_list.append(g['from_idx'] + node_offset)
            to_idx_list.append(g['to_idx'] + node_offset)
            graph_idx_list.append(np.full(n_nodes, i, dtype=np.int32))
            node_offset += n_nodes

        return GraphData(
            from_idx=np.concatenate(from_idx_list) if from_idx_list else np.array([], dtype=np.int32),
            to_idx=np.concatenate(to_idx_list) if to_idx_list else np.array([], dtype=np.int32),
            node_features=np.concatenate(node_features_list),
            edge_features=np.concatenate(edge_features_list) if edge_features_list else np.zeros((0, 1), dtype=np.float32),
            graph_idx=np.concatenate(graph_idx_list),
            n_graphs=len(graphs_list)
        )

    # def pairs(self, batch_size):
    #     while True:
    #         batch_graphs = []
    #         labels = []
    #         for _ in range(batch_size):
    #             is_positive = random.random() > 0.5
    #             if is_positive:
    #                 name = random.choice(self.function_names)
    #                 batch_graphs.extend([self.graphs[name], self.graphs[name]])  # same function twice? No!
    #                 # Wait — we need two different compilations. But since each split is one JSON,
    #                 # we use the same function name for positive (as we did before)
    #                 # Actually for train/val/test we use the same logic:
    #                 batch_graphs.extend([self.graphs[name], self.graphs[name]]) # This is wrong
    #                 # Correction below in full version
    #             else:
    #                 name1 = random.choice(self.function_names)
    #                 name2 = random.choice(self.function_names)
    #                 while name1 == name2:
    #                     name2 = random.choice(self.function_names)
    #                 batch_graphs.extend([self.graphs[name1], self.graphs[name2]])
    #                 labels.append(-1)
    #         # Fix for positive: we need two different versions of same function.
    #         # For now we keep the old logic (positive = same name from same split).
    #         # You can improve later.
    #         yield self._pack_batch(batch_graphs), np.array(labels, dtype=np.int32)

    ## suggested by gemini
    def pairs(self, batch_size):
        while True:
            batch_graphs = []
            labels = []
            for _ in range(batch_size):
                is_positive = random.random() > 0.5
                
                if is_positive:
                    # TODO: We must pull two DIFFERENT compiled versions of 'name'
                    name = random.choice(self.function_names)
                    
                    graph_version_A = self.graphs[name] # e.g., GCC -O0
                    graph_version_B = self.graphs[name] # e.g., Clang -O3 (Currently just a copy!)
                    
                    batch_graphs.extend([graph_version_A, graph_version_B])
                    labels.append(1)  # <--- CRITICAL FIX: Added the positive label!
                    
                else:
                    # Negative Pair: Different functions
                    name1 = random.choice(self.function_names)
                    name2 = random.choice(self.function_names)
                    while name1 == name2:
                        name2 = random.choice(self.function_names)
                        
                    batch_graphs.extend([self.graphs[name1], self.graphs[name2]])
                    labels.append(-1)
                    
            yield self._pack_batch(batch_graphs), np.array(labels, dtype=np.int32)