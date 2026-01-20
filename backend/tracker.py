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
        self.objects = OrderedDict()  # {id: centroid}
        self.disappeared = OrderedDict()  # {id: consecutive_disappeared_frames}
        
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def register(self, centroid):
        """register a new object with next available ID
        
        args:
            centroid: (x, y) coordinates of object center
        
        returns:
            assigned object ID
        """
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        
        return self.next_object_id - 1
    
    def deregister(self, object_id):
        """remove an object from tracking
        
        args:
            object_id: ID of object to remove
        """
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, bboxes):
        """update tracker with new bounding boxes from current frame
        
        args:
            bboxes: list of bounding boxes [(x, y, w, h), ...]
        
        returns:
            dictionary mapping object_id to centroid {id: (cx, cy)}
        """
        # if no bounding boxes, mark all existing objects as disappeared
        if len(bboxes) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                # deregister if disappeared too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            return self.objects
        
        # calculate centroids from bounding boxes
        input_centroids = np.zeros((len(bboxes), 2), dtype="int")
        
        for i, (x, y, w, h) in enumerate(bboxes):
            cx = int(x + w / 2.0)
            cy = int(y + h / 2.0)
            input_centroids[i] = (cx, cy)
        
        # if no objects being tracked, register all
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        
        # otherwise, match existing objects to new centroids
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # compute distance between each pair of object centroids and input centroids
            D = dist.cdist(np.array(object_centroids), input_centroids)
            
            # find smallest value in each row and sort by those values
            rows = D.min(axis=1).argsort()
            
            # find smallest value in each column and sort by row index
            cols = D.argmin(axis=1)[rows]
            
            # track which rows and columns we've already examined
            used_rows = set()
            used_cols = set()
            
            # loop over the combination of (row, column) index tuples
            for (row, col) in zip(rows, cols):
                # ignore if already examined
                if row in used_rows or col in used_cols:
                    continue
                
                # if distance is too large, don't match
                if D[row, col] > self.max_distance:
                    continue
                
                # update centroid and reset disappeared counter
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                used_rows.add(row)
                used_cols.add(col)
            
            # compute row and column indices we haven't examined yet
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            
            # if number of object centroids >= number of input centroids,
            # check if some objects disappeared
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            
            # otherwise, register new input centroids as trackable objects
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])
        
        return self.objects
    
    def get_object_id_by_centroid(self, centroid, threshold=50):
        """get object ID closest to given centroid
        
        args:
            centroid: (x, y) coordinates
            threshold: max distance to consider a match
        
        returns:
            object_id or None if no match found
        """
        if len(self.objects) == 0:
            return None
        
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())
        
        # find closest centroid
        distances = dist.cdist([centroid], object_centroids)[0]
        min_idx = np.argmin(distances)
        
        if distances[min_idx] <= threshold:
            return object_ids[min_idx]
        
        return None
    
    def format_bottle_id(self, object_id):
        """format object ID as bottle ID string
        
        args:
            object_id: numeric object ID
        
        returns:
            formatted bottle ID (e.g., 'BTL_00042')
        """
        return f"BTL_{object_id:05d}"
    
    def reset(self):
        """reset tracker to initial state"""
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()


if __name__ == "__main__":
    # test tracker
    tracker = CentroidTracker()
    
    # frame 1: detect 2 bottles
    bboxes = [(100, 100, 50, 80), (300, 100, 50, 80)]
    objects = tracker.update(bboxes)
    print("frame 1:", {tracker.format_bottle_id(id): c for id, c in objects.items()})
    
    # frame 2: bottles moved slightly
    bboxes = [(105, 100, 50, 80), (305, 100, 50, 80)]
    objects = tracker.update(bboxes)
    print("frame 2:", {tracker.format_bottle_id(id): c for id, c in objects.items()})
    
    # frame 3: one bottle disappeared, one new appeared
    bboxes = [(105, 100, 50, 80), (500, 100, 50, 80)]
    objects = tracker.update(bboxes)
    print("frame 3:", {tracker.format_bottle_id(id): c for id, c in objects.items()})
