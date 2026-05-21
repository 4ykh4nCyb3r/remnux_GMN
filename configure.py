def get_default_config():
    """Optimized configs for real binary function similarity at massive scale."""
    model_type = 'matching'          

    node_state_dim = 64              
    edge_state_dim = 16
    graph_rep_dim = 256              

    graph_embedding_net_config = dict(
        node_state_dim=node_state_dim,
        edge_state_dim=edge_state_dim,
        edge_hidden_sizes=[node_state_dim * 2, node_state_dim * 2],
        node_hidden_sizes=[node_state_dim * 2],
        n_prop_layers=6,  
        share_prop_params=True,
        edge_net_init_scale=0.1,
        node_update_type='gru',       
        use_reverse_direction=True,
        reverse_dir_param_different=False,
        layer_norm=False,              
        prop_type=model_type)

    graph_matching_net_config = graph_embedding_net_config.copy()
    graph_matching_net_config['similarity'] = 'dotproduct'   

    return dict(
        encoder=dict(
            node_hidden_sizes=[node_state_dim],
            node_feature_dim=64,          
            edge_hidden_sizes=[edge_state_dim]),
        aggregator=dict(
            node_hidden_sizes=[graph_rep_dim],
            graph_transform_sizes=[graph_rep_dim],
            input_size=[node_state_dim],
            gated=True,
            aggregation_type='sum'),
        graph_embedding_net=graph_embedding_net_config,
        graph_matching_net=graph_matching_net_config,
        model_type=model_type,
        training=dict(
            # --- UPDATED: GPU Feed Rate ---
            batch_size=64,                # Increased from 16 to feed your GPU faster
            learning_rate=1e-4,
            mode='pair',
            loss='margin',
            margin=1.0,
            graph_vec_regularizer_weight=1e-6,
            clip_value=10.0,
            # --- UPDATED: Scale to new dataset size ---
            n_training_steps=500000,      # Increased from 50k to 500k
            print_after=200,              # Print less often (every 200 steps)
            eval_after=10),               # Evaluate every 2000 steps (200 * 10)
        evaluation=dict(
            batch_size=128),              # Eval takes less RAM, so we can go huge here
        seed=8,
    )