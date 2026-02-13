"""
object tracking module for assigning unique IDs to bottles
uses centroid tracking algorithm
"""
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict


class CentroidTracker:
    """simple centroid-based object tracker for assigning unique IDs"""
    
    def __init__(self, max_disappeared: int = 50, max_distance: int = 50):
        """initialize the centroid tracker
        
        args:
            max_disappeared: max consecutive frames an object can disappear before deregistering
            max_distance: max distance (pixels) to match centroids between frames
        """
        self.next_object_id = 0
        self.objects = OrderedDict()       # {object_id: centroid}
        self.disappeared = OrderedDict()   # {object_id: frames_since_last_seen}
        
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def register(self, centroid) -> int:
        """register a new object with next available ID"""
        object_id = self.next_object_id
        self.objects[object_id] = centroid
        self.disappeared[object_id] = 0
        self.next_object_id += 1
        return object_id
    
    def deregister(self, object_id):
        """remove an object from tracking"""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, bboxes):
        """update tracker with new bounding boxes from current frame
        
        args:
            bboxes: list of bounding boxes [(x, y, w, h), ...]
        
        returns:
            dictionary mapping object_id to centroid {id: (cx, cy)}
        """
        # no detections this frame — age out stale objects
        if len(bboxes) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        # compute centroids from bounding boxes
        input_centroids = np.zeros((len(bboxes), 2), dtype="int")
        for i, (x, y, w, h) in enumerate(bboxes):
            input_centroids[i] = (int(x + w / 2.0), int(y + h / 2.0))
        
        # first frame — register all as new objects
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
            return self.objects
        
        # match existing objects to new detections using pairwise distances
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())
        
        # distance matrix: rows = existing objects, cols = new detections
        distance_matrix = dist.cdist(np.array(object_centroids), input_centroids)
        
        # greedily match by smallest distance first
        sorted_rows = distance_matrix.min(axis=1).argsort()
        matched_cols = distance_matrix.argmin(axis=1)[sorted_rows]
        
        used_rows = set()
        used_cols = set()
        
        for row, col in zip(sorted_rows, matched_cols):
            if row in used_rows or col in used_cols:
                continue
            if distance_matrix[row, col] > self.max_distance:
                continue
            
            # update matched object with new centroid
            self.objects[object_ids[row]] = input_centroids[col]
            self.disappeared[object_ids[row]] = 0
            used_rows.add(row)
            used_cols.add(col)
        
        unused_rows = set(range(distance_matrix.shape[0])) - used_rows
        unused_cols = set(range(distance_matrix.shape[1])) - used_cols
        
        if distance_matrix.shape[0] >= distance_matrix.shape[1]:
            # more existing objects than detections — age unmatched objects
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
        else:
            # more detections than existing objects — register new ones
            for col in unused_cols:
                self.register(input_centroids[col])
        
        return self.objects
    
    def get_object_id_by_centroid(self, centroid, threshold=50):
        """find the tracked object closest to the given centroid
        
        returns:
            object_id if within threshold, else None
        """
        if not self.objects:
            return None
        
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())
        
        distances = dist.cdist([centroid], object_centroids)[0]
        min_idx = np.argmin(distances)
        
        if distances[min_idx] <= threshold:
            return object_ids[min_idx]
        return None
    
    def format_bottle_id(self, object_id) -> str:
        """format numeric object ID as bottle ID string (e.g. 'BTL_00042')"""
        return f"BTL_{object_id:05d}"
    
    def reset(self):
        """reset tracker to initial state"""
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
