"""
Vision Module - Pose detection, skin analysis, and zone guarding detection
Improvements:
- Better angle calculation using numpy (numerically more stable)
- Documented line intersection algorithm
"""
import math
import numpy as np


def calculate_angle(a, b, c):
    """
    Calculate angle at vertex b formed by points a-b-c using dot product.
    More numerically stable than atan2 method.
    
    Args:
        a, b, c: Landmarks with .x, .y attributes (MediaPipe format)
    
    Returns:
        float: Angle in degrees (0-180)
    """
    a_vec = np.array([a.x, a.y])
    b_vec = np.array([b.x, b.y])
    c_vec = np.array([c.x, c.y])
    
    ba = a_vec - b_vec
    bc = c_vec - b_vec
    
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    cos_angle = np.dot(ba, bc) / denom
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle) * 180 / np.pi
    
    return angle


def ccw(A, B, C):
    """
    Check if three points are in counter-clockwise order.
    Used for line segment intersection detection.
    """
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def check_intersection(A, B, C, D):
    """
    Check if line segments AB and CD intersect using CCW orientation test.
    
    Args:
        A, B: Endpoints of first line segment (tuples)
        C, D: Endpoints of second line segment (tuples)
    
    Returns:
        bool: True if segments intersect (not including collinear cases)
    
    Note:
        This implementation uses CCW orientation test.
        Does not handle collinear segments specially.
    """
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
