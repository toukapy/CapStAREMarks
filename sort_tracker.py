# sort_tracker.py
# Pure-Python SORT implementation (no C++ / no compilation)

import numpy as np
from filterpy.kalman import KalmanFilter

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
              + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return o


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = bbox.reshape((4,1))
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(bbox.reshape((4,1)))

    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.kf.x[:4].reshape((4,))


class Sort:
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, dets):
        trks = np.zeros((len(self.trackers), 5))
        to_del = []

        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t][:4] = pos
            trks[t][4] = trk.id
            if np.any(np.isnan(pos)):
                to_del.append(t)

        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])

        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i, :4]))

        ret = []
        for trk in self.trackers:
            if trk.time_since_update < 1 and (trk.hits >= self.min_hits):
                d = trk.kf.x[:4].reshape((4,))
                ret.append(np.concatenate((d, [trk.id])))

        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        return np.array(ret)


def associate_detections_to_trackers(detections, trackers, iou_threshold):
    if len(trackers) == 0:
        return np.empty((0,2), dtype=int), np.arange(len(detections)), np.empty((0), dtype=int)

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det[:4], trk[:4])

    matched_indices = np.array(np.where(iou_matrix > iou_threshold)).T

    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[:,0]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[:,1]:
            unmatched_trackers.append(t)

    matches = matched_indices
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
