"""
Utility functions for the CyberThreat-ML library.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from .logger import logger

def save_dataset(X, y, dataset_path, metadata=None):
    """
    Save a dataset (features and labels) to disk.
    
    Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Labels.
        dataset_path (str): Path to save the dataset.
        metadata (dict, optional): Dataset metadata.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    
    # Save data and labels
    np.savez_compressed(
        dataset_path,
        X=X,
        y=y,
        metadata=metadata or {}
    )
    
    logger.info(f"Dataset saved to {dataset_path}, X shape: {X.shape}, y shape: {y.shape}")

def load_dataset(dataset_path):
    """
    Load a dataset from disk.
    
    Args:
        dataset_path (str): Path to the saved dataset.
        
    Returns:
        tuple: (X, y, metadata) - features, labels, and metadata.
    """
    # Load the dataset
    data = np.load(dataset_path, allow_pickle=True)
    
    X = data['X']
    y = data['y']
    metadata = data.get('metadata', None)
    
    if metadata is not None and isinstance(metadata, np.ndarray):
        metadata = metadata.item()
    
    logger.info(f"Dataset loaded from {dataset_path}, X shape: {X.shape}, y shape: {y.shape}")
    return X, y, metadata

def split_data(X, y, test_size=0.2, val_size=0.25, random_state=None):
    """
    Split data into training, validation, and test sets.
    
    Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Labels.
        test_size (float): Proportion of data to use for testing.
        val_size (float): Proportion of training data to use for validation.
        random_state (int, optional): Random seed for reproducibility.
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: training+validation and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state
    )
    
    logger.info(f"Data split: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def normalize_packet_size(size, max_size=65535):
    """
    Normalize packet size to range [0, 1].
    
    Args:
        size (int): Packet size in bytes.
        max_size (int): Maximum possible packet size.
        
    Returns:
        float: Normalized packet size.
    """
    return min(size, max_size) / max_size

def normalize_port_number(port):
    """
    Normalize port number to range [0, 1].
    
    Args:
        port (int): Port number.
        
    Returns:
        float: Normalized port number.
    """
    return port / 65535  # Maximum port number

def normalize_timestamp(timestamp, reference_time=None):
    """
    Normalize timestamp to seconds since reference time.
    
    Args:
        timestamp (float): Unix timestamp.
        reference_time (float, optional): Reference time (if None, uses timestamp).
        
    Returns:
        float: Normalized timestamp (seconds since reference).
    """
    ref_time = reference_time if reference_time is not None else timestamp
    return timestamp - ref_time

def calculate_entropy(data):
    """
    Calculate Shannon entropy of data.
    
    Args:
        data (bytes or list): Input data.
        
    Returns:
        float: Entropy value.
    """
    if not data:
        return 0
    
    # Count occurrences of each value
    counter = {}
    for value in data:
        counter[value] = counter.get(value, 0) + 1
    
    # Calculate entropy
    entropy = 0
    total = len(data)
    
    for count in counter.values():
        probability = count / total
        entropy -= probability * np.log2(probability)
    
    return entropy

def plot_training_history(history, figsize=(12, 4)):
    """
    Plot the training history of a model.
    
    Args:
        history (tensorflow.keras.callbacks.History): Training history.
        figsize (tuple): Figure size.
        
    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot loss
    ax1.plot(history.history['loss'], label='train')
    if 'val_loss' in history.history:
        ax1.plot(history.history['val_loss'], label='validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(history.history['accuracy'], label='train')
    if 'val_accuracy' in history.history:
        ax2.plot(history.history['val_accuracy'], label='validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def parse_pcap_file(file_path, max_packets=None):
    """
    Parse a PCAP file and extract packet features.
    
    Args:
        file_path (str): Path to the PCAP file.
        max_packets (int, optional): Maximum number of packets to parse.
        
    Returns:
        pandas.DataFrame: Extracted packet features.
    """
    try:
        import dpkt
        import socket
        from dpkt.ethernet import Ethernet
        
        # List to store packet features
        packets = []
        counter = 0
        
        # Open and parse PCAP file
        with open(file_path, 'rb') as f:
            pcap = dpkt.pcap.Reader(f)
            
            for timestamp, buf in pcap:
                try:
                    # Parse Ethernet packet
                    eth = Ethernet(buf)
                    
                    # Basic packet information
                    packet_info = {
                        'timestamp': timestamp
                    }
                    
                    # Check if it's an IP packet
                    if isinstance(eth.data, dpkt.ip.IP):
                        ip = eth.data
                        
                        # Extract IP information
                        packet_info.update({
                            'src_ip': socket.inet_ntoa(ip.src),
                            'dst_ip': socket.inet_ntoa(ip.dst),
                            'protocol': ip.p,
                            'packet_len': ip.len,
                            'ttl': ip.ttl
                        })
                        
                        # Extract TCP/UDP information
                        if isinstance(ip.data, dpkt.tcp.TCP):
                            tcp = ip.data
                            packet_info.update({
                                'src_port': tcp.sport,
                                'dst_port': tcp.dport,
                                'tcp_flags': int(tcp.flags),
                                'payload_len': len(tcp.data),
                                'window_size': tcp.win
                            })
                        elif isinstance(ip.data, dpkt.udp.UDP):
                            udp = ip.data
                            packet_info.update({
                                'src_port': udp.sport,
                                'dst_port': udp.dport,
                                'payload_len': len(udp.data)
                            })
                    
                    packets.append(packet_info)
                    counter += 1
                    
                    if max_packets is not None and counter >= max_packets:
                        break
                
                except Exception as e:
                    logger.warning(f"Error parsing packet: {str(e)}")
                    continue
        
        logger.info(f"Parsed {len(packets)} packets from {file_path}")
        
        # Convert to DataFrame
        return pd.DataFrame(packets)
    
    except ImportError:
        logger.error("dpkt module not found. Please install it with: pip install dpkt")
        raise
    except Exception as e:
        logger.error(f"Error parsing PCAP file: {str(e)}")
        raise
