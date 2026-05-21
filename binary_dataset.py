import sqlite3
import json
import numpy as np
import collections
import random

GraphData = collections.namedtuple('GraphData', [
    'from_idx', 'to_idx', 'node_features', 'edge_features', 'graph_idx', 'n_graphs'])

class BinaryDataset:
    def __init__(self, db_path):
        self.db_path = db_path
        
        # Connect to DB just to get the valid function names
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        print(f"[*] Scanning database {db_path}...")
        
        # Find all functions that have AT LEAST 2 versions (so we can make positive pairs)
        cursor.execute('''
            SELECT function_name 
            FROM graphs 
            GROUP BY function_name 
            HAVING COUNT(*) >= 2
        ''')
        
        # Save just the names in RAM (takes almost 0 memory)
        self.valid_function_names = [row[0] for row in cursor.fetchall()]
        
        print(f"[*] Found {len(self.valid_function_names)} unique functions capable of forming pairs.")
        conn.close()
    
    """
    parsed = {
    'main': [
        {
            'node_features': array([[1.0, 0.5], [2.0, 1.5], [3.0, 2.5]], dtype=float32),
            'edge_features': array([[1.0], [1.0]], dtype=float32),
            'from_idx': array([0, 1], dtype=int32),
            'to_idx': array([1, 2], dtype=int32)
        },  # GCC compiled version
        {
            'node_features': array([[1.1, 0.6], [2.1, 1.6]], dtype=float32),
            'edge_features': array([[1.0]], dtype=float32),
            'from_idx': array([0], dtype=int32),
            'to_idx': array([1], dtype=int32)
        }   # Clang compiled version
    ],
    'helper': [
        {
            'node_features': array([[5.0, 3.0]], dtype=float32),
            'edge_features': array([], shape=(0, 1), dtype=float32),
            'from_idx': array([], dtype=int32),
            'to_idx': array([], dtype=int32)
        }   # No edges
    ]
    }

    """

    # def _pack_batch(self, graphs_list):
    #     from_idx_list, to_idx_list, node_features_list, edge_features_list, graph_idx_list = [], [], [], [], []
    #     node_offset = 0
    #     for i, g in enumerate(graphs_list):
    #         n_nodes = g['node_features'].shape[0]
    #         node_features_list.append(g['node_features'])
    #         edge_features_list.append(g['edge_features'])
    #         from_idx_list.append(g['from_idx'] + node_offset)
    #         to_idx_list.append(g['to_idx'] + node_offset)
    #         graph_idx_list.append(np.full(n_nodes, i, dtype=np.int32))
    #         node_offset += n_nodes

    #     return GraphData(
    #         from_idx=np.concatenate(from_idx_list) if from_idx_list else np.array([], dtype=np.int32),
    #         to_idx=np.concatenate(to_idx_list) if to_idx_list else np.array([], dtype=np.int32),
    #         node_features=np.concatenate(node_features_list),
    #         edge_features=np.concatenate(edge_features_list) if edge_features_list else np.zeros((0, 1), dtype=np.float32),
    #         graph_idx=np.concatenate(graph_idx_list),
    #         n_graphs=len(graphs_list)
    #     )

    def _pack_batch(self, graphs_list):
        from_idx_list, to_idx_list, node_features_list, edge_features_list, graph_idx_list = [], [], [], [], []
        node_offset = 0
        
        for i, g in enumerate(graphs_list):
            # --- THE FIX: Convert Python lists into NumPy arrays on the fly ---
            n_feat = np.array(g['node_features'], dtype=np.float32)
            
            # Handle edge cases safely (if a tiny function has no edges)
            e_feat = np.array(g['edge_features'], dtype=np.float32) if len(g.get('edge_features', [])) > 0 else np.zeros((0, 1), dtype=np.float32)
            f_idx = np.array(g['from_idx'], dtype=np.int32) if len(g.get('from_idx', [])) > 0 else np.array([], dtype=np.int32)
            t_idx = np.array(g['to_idx'], dtype=np.int32) if len(g.get('to_idx', [])) > 0 else np.array([], dtype=np.int32)

            n_nodes = n_feat.shape[0]
            
            node_features_list.append(n_feat)
            edge_features_list.append(e_feat)
            
            # Now math like + node_offset works perfectly because they are NumPy arrays!
            from_idx_list.append(f_idx + node_offset)
            to_idx_list.append(t_idx + node_offset)
            
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

    def pairs(self, batch_size):
        # We open the connection INSIDE the generator so it plays nicely with PyTorch workers
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        while True:
            batch_graphs = []
            labels = []
            
            for _ in range(batch_size):
                is_positive = random.random() > 0.5
                
                if is_positive:
                    # MATCH
                    name = random.choice(self.valid_function_names)
                    
                    # Ask SQLite to give us 2 random versions of this specific function
                    cursor.execute('''
                        SELECT graph_data FROM graphs 
                        WHERE function_name = ? 
                        ORDER BY RANDOM() LIMIT 2
                    ''', (name,))
                    
                    rows = cursor.fetchall()
                    version_A = json.loads(rows[0][0])
                    version_B = json.loads(rows[1][0])
                    
                    batch_graphs.extend([version_A, version_B])
                    labels.append(1)
                    
                else:
                    # MISMATCH
                    name1, name2 = random.sample(self.valid_function_names, 2)
                    
                    # Ask SQLite for 1 random version of name1
                    cursor.execute('SELECT graph_data FROM graphs WHERE function_name = ? ORDER BY RANDOM() LIMIT 1', (name1,))
                    version_A = json.loads(cursor.fetchone()[0])
                    
                    # Ask SQLite for 1 random version of name2
                    cursor.execute('SELECT graph_data FROM graphs WHERE function_name = ? ORDER BY RANDOM() LIMIT 1', (name2,))
                    version_B = json.loads(cursor.fetchone()[0])
                    
                    batch_graphs.extend([version_A, version_B])
                    labels.append(-1)
                    
            yield self._pack_batch(batch_graphs), np.array(labels, dtype=np.int32)