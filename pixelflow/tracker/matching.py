"""
Matching utilities for object tracking in PixelFlow.

This module provides functions for matching detections to tracks using
IoU (Intersection over Union) and Hungarian algorithm.
"""

import numpy as np
from typing import List, Tuple, Optional
import scipy.optimize


def box_iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes_a: Array of boxes in format [x1, y1, x2, y2] with shape (N, 4)
        boxes_b: Array of boxes in format [x1, y1, x2, y2] with shape (M, 4)
    
    Returns:
        IoU matrix with shape (N, M)
    """
    def box_area(box):
        return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    
    area_a = box_area(boxes_a)
    area_b = box_area(boxes_b)
    
    lt = np.maximum(boxes_a[:, None, :2], boxes_b[:, :2])  # [N, M, 2]
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[:, 2:])  # [N, M, 2]
    
    wh = np.clip(rb - lt, 0, None)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    union = area_a[:, None] + area_b - inter
    
    iou = inter / (union + 1e-7)
    return iou


def iou_distance(tracks_a: List, tracks_b: List) -> np.ndarray:
    """
    Compute IoU distance between two lists of tracks/detections.
    
    Args:
        tracks_a: List of tracks or detections with tlbr property
        tracks_b: List of tracks or detections with tlbr property
    
    Returns:
        Cost matrix based on IoU distance (1 - IoU)
    """
    if len(tracks_a) == 0 or len(tracks_b) == 0:
        return np.zeros((len(tracks_a), len(tracks_b)))
    
    boxes_a = np.array([track.tlbr for track in tracks_a])
    boxes_b = np.array([track.tlbr for track in tracks_b])
    
    ious = box_iou_batch(boxes_a, boxes_b)
    cost_matrix = 1 - ious
    
    return cost_matrix


def embedding_distance(tracks_a: List, tracks_b: List, metric: str = 'cosine') -> np.ndarray:
    """
    Compute embedding distance between tracks (for future ReID integration).
    
    Args:
        tracks_a: List of tracks with embedding features
        tracks_b: List of tracks with embedding features
        metric: Distance metric ('cosine' or 'euclidean')
    
    Returns:
        Distance matrix
    """
    # Placeholder for future ReID feature integration
    # For now, return zeros (no embedding distance)
    return np.zeros((len(tracks_a), len(tracks_b)))


def fuse_score(cost_matrix: np.ndarray, detections: List, weight: float = 1.4) -> np.ndarray:
    """
    Fuse detection scores with cost matrix for better matching.
    
    Args:
        cost_matrix: Initial cost matrix from IoU or other distance
        detections: List of detections with score attribute
        weight: Weight factor for score fusion
    
    Returns:
        Fused cost matrix
    """
    if cost_matrix.shape[1] == 0:
        return cost_matrix
    
    # Get detection scores
    det_scores = np.array([det.score for det in detections])
    
    # Expand scores to match cost matrix shape
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    
    # Fuse scores with cost matrix
    # Higher detection scores should reduce the cost
    fuse_sim = 1 - cost_matrix
    fuse_sim = fuse_sim * (1 + det_scores * weight)
    fused_cost = 1 - fuse_sim
    
    return fused_cost


def linear_assignment(cost_matrix: np.ndarray, thresh: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve linear assignment problem using Hungarian algorithm.
    
    Args:
        cost_matrix: Cost matrix for assignment
        thresh: Threshold for valid assignments
    
    Returns:
        Tuple of (matches, unmatched_a, unmatched_b)
        - matches: Array of matched indices pairs
        - unmatched_a: Indices from first set that weren't matched
        - unmatched_b: Indices from second set that weren't matched
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), np.arange(cost_matrix.shape[0]), np.arange(cost_matrix.shape[1])
    
    # Solve assignment problem
    row_indices, col_indices = scipy.optimize.linear_sum_assignment(cost_matrix)
    
    matches = []
    unmatched_a = []
    unmatched_b = []
    
    # Filter matches by threshold
    for i, j in zip(row_indices, col_indices):
        if cost_matrix[i, j] <= thresh:
            matches.append([i, j])
        else:
            unmatched_a.append(i)
            unmatched_b.append(j)
    
    # Find completely unmatched indices
    for i in range(cost_matrix.shape[0]):
        if i not in row_indices:
            unmatched_a.append(i)
    
    for j in range(cost_matrix.shape[1]):
        if j not in col_indices:
            unmatched_b.append(j)
    
    matches = np.array(matches) if matches else np.empty((0, 2), dtype=int)
    unmatched_a = np.array(unmatched_a)
    unmatched_b = np.array(unmatched_b)
    
    return matches, unmatched_a, unmatched_b


def min_cost_matching(distance_matrix: np.ndarray, max_distance: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve minimum cost matching problem.
    
    This is an alternative to linear_assignment that can handle
    different distance metrics.
    
    Args:
        distance_matrix: Distance matrix for matching
        max_distance: Maximum allowed distance for valid match
    
    Returns:
        Tuple of (matches, unmatched_a, unmatched_b)
    """
    # Set distances above threshold to infinity
    cost_matrix = np.copy(distance_matrix)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    
    return linear_assignment(cost_matrix, max_distance)