# global_vars.py
import numpy as np
import os

# Initialize global variables with specific data types for memory efficiency
body_pose = np.zeros(3, dtype=np.float32)  # x, y, theta
lidar_data = np.zeros(20, dtype=np.float32)
clash_sum = 0
image = None
image_for_clash_calc = None

# Memory management section - used by memory-mapped arrays
_mmap_files = []

def register_mmap_file(filepath):
    """Register a memory-mapped file for cleanup"""
    global _mmap_files
    _mmap_files.append(filepath)

def cleanup_mmap_files():
    """Clean up all registered memory-mapped files"""
    global _mmap_files
    for filepath in _mmap_files:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            print(f"Error cleaning up memory-mapped file {filepath}: {e}")
    
    _mmap_files = []

# Register cleaning function to be called at exit
import atexit
atexit.register(cleanup_mmap_files)