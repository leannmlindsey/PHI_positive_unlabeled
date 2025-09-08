"""
Graph-based community detection for data splitting
Prevents data leakage by keeping similar proteins in same split
"""

import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import community
from collections import defaultdict
import hashlib
from typing import Tuple, List, Dict, Set
import pickle
from sklearn.model_selection import train_test_split

class GraphBasedDataSplitter:
    """
    Splits phage-host interaction data using graph-based community detection
    to prevent data leakage from shared proteins
    """
    
    def __init__(self, data_path: str, seed: int = 42):
        self.data_path = data_path
        self.seed = seed
        np.random.seed(seed)
        
        # Load data
        self.df = pd.read_csv(data_path, sep='\t')
        print(f"Loaded {len(self.df)} interactions")
        
        # Track unique proteins
        self.marker_to_samples = defaultdict(set)
        self.rbp_to_samples = defaultdict(set)
        self.sample_to_markers = {}
        self.sample_to_rbps = {}
        
    def parse_proteins(self):
        """Extract and hash all proteins"""
        print("Parsing protein sequences...")
        
        for idx, row in self.df.iterrows():
            # Parse marker proteins (host)
            marker_seqs = row['marker_gene_seq'].split(',')
            marker_hashes = row['marker_md5'].split(',') if ',' in str(row['marker_md5']) else [row['marker_md5']]
            
            # Parse RBP proteins (phage)
            rbp_seqs = row['rbp_seq'].split(',')
            rbp_hashes = row['rbp_md5'].split(',') if ',' in str(row['rbp_md5']) else [row['rbp_md5']]
            
            # Store mappings
            self.sample_to_markers[idx] = marker_hashes
            self.sample_to_rbps[idx] = rbp_hashes
            
            # Track which samples contain each protein
            for m_hash in marker_hashes:
                self.marker_to_samples[m_hash].add(idx)
            for r_hash in rbp_hashes:
                self.rbp_to_samples[r_hash].add(idx)
                
        print(f"Found {len(self.marker_to_samples)} unique marker proteins")
        print(f"Found {len(self.rbp_to_samples)} unique RBP proteins")
        
        # Find shared proteins
        shared_markers = sum(1 for samples in self.marker_to_samples.values() if len(samples) > 1)
        shared_rbps = sum(1 for samples in self.rbp_to_samples.values() if len(samples) > 1)
        print(f"Markers appearing in multiple samples: {shared_markers}")
        print(f"RBPs appearing in multiple samples: {shared_rbps}")
        
    def build_interaction_graph(self) -> nx.Graph:
        """
        Build a graph where:
        - Nodes are samples (interactions)
        - Edges connect samples that share proteins
        - Edge weights indicate strength of connection
        """
        print("Building interaction graph...")
        G = nx.Graph()
        
        # Add all samples as nodes
        G.add_nodes_from(range(len(self.df)))
        
        # Connect samples that share proteins
        edges_added = 0
        
        # Connect via shared markers
        for marker_hash, sample_ids in self.marker_to_samples.items():
            if len(sample_ids) > 1:
                # Connect all pairs of samples with this marker
                sample_list = list(sample_ids)
                for i in range(len(sample_list)):
                    for j in range(i+1, len(sample_list)):
                        if G.has_edge(sample_list[i], sample_list[j]):
                            # Increase weight if edge exists
                            G[sample_list[i]][sample_list[j]]['weight'] += 1.0
                            G[sample_list[i]][sample_list[j]]['shared_markers'].add(marker_hash)
                        else:
                            G.add_edge(sample_list[i], sample_list[j], 
                                     weight=1.0,
                                     shared_markers={marker_hash},
                                     shared_rbps=set())
                            edges_added += 1
        
        # Connect via shared RBPs
        for rbp_hash, sample_ids in self.rbp_to_samples.items():
            if len(sample_ids) > 1:
                sample_list = list(sample_ids)
                for i in range(len(sample_list)):
                    for j in range(i+1, len(sample_list)):
                        if G.has_edge(sample_list[i], sample_list[j]):
                            G[sample_list[i]][sample_list[j]]['weight'] += 2.0  # RBPs weighted more
                            G[sample_list[i]][sample_list[j]]['shared_rbps'].add(rbp_hash)
                        else:
                            G.add_edge(sample_list[i], sample_list[j], 
                                     weight=2.0,  # RBPs weighted more heavily
                                     shared_markers=set(),
                                     shared_rbps={rbp_hash})
                            edges_added += 1
        
        print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Analyze connectivity
        components = list(nx.connected_components(G))
        print(f"Number of connected components: {len(components)}")
        component_sizes = [len(c) for c in components]
        print(f"Component sizes - min: {min(component_sizes)}, max: {max(component_sizes)}, "
              f"mean: {np.mean(component_sizes):.1f}")
        
        return G
    
    def detect_communities(self, G: nx.Graph) -> List[Set[int]]:
        """
        Detect communities using Louvain algorithm
        Communities represent groups of interactions that share proteins
        """
        print("Detecting communities...")
        
        # Use Louvain method for community detection
        communities = community.louvain_communities(G, weight='weight', seed=self.seed)
        
        print(f"Found {len(communities)} communities")
        community_sizes = [len(c) for c in communities]
        print(f"Community sizes - min: {min(community_sizes)}, max: {max(community_sizes)}, "
              f"mean: {np.mean(community_sizes):.1f}")
        
        return communities
    
    def split_communities(self, communities: List[Set[int]], 
                         train_ratio: float = 0.6,
                         val_ratio: float = 0.2) -> Tuple[List[int], List[int], List[int]]:
        """
        Split communities into train/val/test sets
        Ensures entire communities stay together to prevent leakage
        """
        print(f"\nSplitting communities (train={train_ratio}, val={val_ratio}, test={1-train_ratio-val_ratio})...")
        
        # Sort communities by size for stratified splitting
        sorted_communities = sorted(communities, key=len, reverse=True)
        
        # Calculate target sizes
        total_samples = sum(len(c) for c in sorted_communities)
        target_train = int(total_samples * train_ratio)
        target_val = int(total_samples * val_ratio)
        
        train_communities = []
        val_communities = []
        test_communities = []
        
        train_size = 0
        val_size = 0
        test_size = 0
        
        # Assign communities to splits
        for community in sorted_communities:
            comm_size = len(community)
            
            # Assign to the split that needs samples most
            train_diff = abs((train_size + comm_size) - target_train)
            val_diff = abs((val_size + comm_size) - target_val)
            test_diff = abs((test_size + comm_size) - (total_samples - target_train - target_val))
            
            if train_size < target_train and train_diff <= min(val_diff, test_diff):
                train_communities.append(community)
                train_size += comm_size
            elif val_size < target_val and val_diff <= test_diff:
                val_communities.append(community)
                val_size += comm_size
            else:
                test_communities.append(community)
                test_size += comm_size
        
        # Flatten communities to sample indices
        train_idx = [idx for comm in train_communities for idx in comm]
        val_idx = [idx for comm in val_communities for idx in comm]
        test_idx = [idx for comm in test_communities for idx in comm]
        
        print(f"Final split sizes - Train: {len(train_idx)} ({len(train_idx)/total_samples:.1%}), "
              f"Val: {len(val_idx)} ({len(val_idx)/total_samples:.1%}), "
              f"Test: {len(test_idx)} ({len(test_idx)/total_samples:.1%})")
        
        print(f"Communities per split - Train: {len(train_communities)}, "
              f"Val: {len(val_communities)}, Test: {len(test_communities)}")
        
        return train_idx, val_idx, test_idx
    
    def verify_no_leakage(self, train_idx: List[int], val_idx: List[int], test_idx: List[int]):
        """
        Verify that no proteins are shared between splits
        This is the critical check to ensure no data leakage
        """
        print("\nVerifying no data leakage...")
        
        # Get proteins in each split
        train_markers = set()
        train_rbps = set()
        for idx in train_idx:
            train_markers.update(self.sample_to_markers[idx])
            train_rbps.update(self.sample_to_rbps[idx])
        
        val_markers = set()
        val_rbps = set()
        for idx in val_idx:
            val_markers.update(self.sample_to_markers[idx])
            val_rbps.update(self.sample_to_rbps[idx])
        
        test_markers = set()
        test_rbps = set()
        for idx in test_idx:
            test_markers.update(self.sample_to_markers[idx])
            test_rbps.update(self.sample_to_rbps[idx])
        
        # Check for overlaps
        marker_train_val = train_markers & val_markers
        marker_train_test = train_markers & test_markers
        marker_val_test = val_markers & test_markers
        
        rbp_train_val = train_rbps & val_rbps
        rbp_train_test = train_rbps & test_rbps
        rbp_val_test = val_rbps & test_rbps
        
        print(f"\nMarker overlaps:")
        print(f"  Train-Val: {len(marker_train_val)} shared markers")
        print(f"  Train-Test: {len(marker_train_test)} shared markers")
        print(f"  Val-Test: {len(marker_val_test)} shared markers")
        
        print(f"\nRBP overlaps:")
        print(f"  Train-Val: {len(rbp_train_val)} shared RBPs")
        print(f"  Train-Test: {len(rbp_train_test)} shared RBPs")
        print(f"  Val-Test: {len(rbp_val_test)} shared RBPs")
        
        # Calculate coverage
        total_unique_markers = len(train_markers | val_markers | test_markers)
        total_unique_rbps = len(train_rbps | val_rbps | test_rbps)
        
        print(f"\nProtein coverage:")
        print(f"  Train: {len(train_markers)} markers, {len(train_rbps)} RBPs")
        print(f"  Val: {len(val_markers)} markers, {len(val_rbps)} RBPs")
        print(f"  Test: {len(test_markers)} markers, {len(test_rbps)} RBPs")
        print(f"  Total unique: {total_unique_markers} markers, {total_unique_rbps} RBPs")
        
        # Return whether there's leakage
        has_leakage = (len(marker_train_val) > 0 or len(marker_train_test) > 0 or 
                      len(marker_val_test) > 0 or len(rbp_train_val) > 0 or 
                      len(rbp_train_test) > 0 or len(rbp_val_test) > 0)
        
        if has_leakage:
            print("\n⚠️ WARNING: Data leakage detected! Proteins are shared between splits.")
        else:
            print("\n✅ No data leakage detected! All splits have unique proteins.")
        
        return not has_leakage
    
    def save_splits(self, train_idx, val_idx, test_idx, output_dir='data/processed'):
        """Save the split indices and data"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save indices
        splits = {
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx
        }
        
        with open(f'{output_dir}/splits.pkl', 'wb') as f:
            pickle.dump(splits, f)
        
        # Save actual data splits
        train_df = self.df.iloc[train_idx].copy()
        val_df = self.df.iloc[val_idx].copy()
        test_df = self.df.iloc[test_idx].copy()
        
        train_df.to_csv(f'{output_dir}/train.tsv', sep='\t', index=False)
        val_df.to_csv(f'{output_dir}/val.tsv', sep='\t', index=False)
        test_df.to_csv(f'{output_dir}/test.tsv', sep='\t', index=False)
        
        print(f"\nSplits saved to {output_dir}/")
        
    def run(self):
        """Execute the full splitting pipeline"""
        # Parse proteins and build mappings
        self.parse_proteins()
        
        # Build interaction graph
        G = self.build_interaction_graph()
        
        # Detect communities
        communities = self.detect_communities(G)
        
        # Split communities
        train_idx, val_idx, test_idx = self.split_communities(communities)
        
        # Verify no leakage
        no_leakage = self.verify_no_leakage(train_idx, val_idx, test_idx)
        
        if no_leakage:
            # Save splits
            self.save_splits(train_idx, val_idx, test_idx)
        else:
            print("\n⚠️ Splits not saved due to data leakage. Adjusting strategy...")
            # Could implement fallback strategy here
        
        return train_idx, val_idx, test_idx


if __name__ == "__main__":
    splitter = GraphBasedDataSplitter('data/dedup.phage_marker_rbp_with_phage_entropy.tsv')
    train_idx, val_idx, test_idx = splitter.run()