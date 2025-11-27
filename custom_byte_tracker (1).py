# /root/autodl-tmp/custom_byte_tracker.py
import numpy as np
from collections import deque
from filterpy.kalman import KalmanFilter as FilterPyKalmanFilter
from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH
from ultralytics.trackers.basetrack import BaseTrack, TrackState
from ultralytics.utils.checks import check_requirements
from ultralytics.trackers.utils import matching

check_requirements('lap')
import lap

# ... (motion_mahalanobis_distance 和 mahalanobis_distance_depth 保持不变) ...
def motion_mahalanobis_distance(tracks, detections):
    """计算航迹和检测框之间的2D运动(xyah)马氏距离"""
    cost_matrix = np.full((len(tracks), len(detections)), np.inf, dtype=np.float32)
    if len(tracks) == 0 or len(detections) == 0:
        return cost_matrix

    kf_motion = STrack.shared_kalman
    multi_track_mean = np.asarray([st.mean for st in tracks])
    multi_track_cov = np.asarray([st.covariance for st in tracks])
    multi_det_mean = np.asarray([STrack.tlwh_to_xyah(det.tlwh) for det in detections])

    for i, (track_mean, track_cov) in enumerate(zip(multi_track_mean, multi_track_cov)):
        projected_mean, projected_cov = kf_motion.project(track_mean, track_cov)
        try:
            cholesky_factor = np.linalg.cholesky(projected_cov)
            inv_projected_cov = np.linalg.inv(cholesky_factor).T @ np.linalg.inv(cholesky_factor)
            for j, det_mean in enumerate(multi_det_mean):
                residual = det_mean - projected_mean
                motion_dist_sq = residual @ inv_projected_cov @ residual.T
                cost_matrix[i, j] = motion_dist_sq
        except np.linalg.LinAlgError:
            continue
    return cost_matrix

def mahalanobis_distance_depth(tracks, detections):
    """计算航迹和检测框之间的一维深度马氏距离"""
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if len(tracks) == 0 or len(detections) == 0:
        return cost_matrix

    for i, track in enumerate(tracks):
        kf = track.kf_depth
        predicted_state = kf.x
        predicted_covariance = kf.P
        measurement_matrix = kf.H
        measurement_noise = kf.R
        innovation_covariance = measurement_matrix @ predicted_covariance @ measurement_matrix.T + measurement_noise
        for j, det in enumerate(detections):
            measurement = np.array([det.depth])
            residual = measurement - measurement_matrix @ predicted_state
            maha_dist_sq = residual.T @ np.linalg.inv(innovation_covariance) @ residual
            cost_matrix[i, j] = maha_dist_sq
    return cost_matrix


class STrack(BaseTrack):
    shared_kalman = KalmanFilterXYAH()

    # [修改] 构造函数接收 KF 参数
    def __init__(self, tlwh, score, cls, initial_depth=0.0, 
                 kf_R=5.0, kf_Q_pos=0.1, kf_Q_vel=0.01):
        super().__init__()
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.kalman_filter = None; self.mean, self.covariance = None, None
        self.is_activated = False; self.score = score
        self.tracklet_len = 0; self.cls = cls; self.idx = 0
        self.depth = initial_depth
        # [修改] 将 KF 参数传递给初始化函数
        self.kf_depth = self.init_depth_kalman_filter(initial_depth, kf_R, kf_Q_pos, kf_Q_vel)
        self.angle = None
    
    # ... (release_id 和 xyxy 保持不变) ...
    @staticmethod
    def release_id():
        """
        重置父类 BaseTrack 的ID计数器，确保每个视频的ID从1开始。
        """
        BaseTrack._count = 0

    @property
    def xyxy(self): return self.tlbr

    # [修改] 初始化函数接收 KF 参数
    def init_depth_kalman_filter(self, initial_depth, kf_R, kf_Q_pos, kf_Q_vel):
        kf = FilterPyKalmanFilter(dim_x=2, dim_z=1); kf.x = np.array([initial_depth, 0.])
        kf.F = np.array([[1., 1.], [0., 1.]]); kf.H = np.array([[1., 0.]]); kf.P *= 100.
        # [修改] 使用传入的参数，而不是硬编码
        kf.R = np.array([[kf_R]])
        kf.Q = np.diag([kf_Q_pos, kf_Q_vel])
        return kf

    # ... (predict, multi_predict, activate, re_activate, update, tlwh, tlbr, ... 保持不变) ...
    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked: mean_state[6] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
        self.kf_depth.predict()
    
    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked: multi_mean[i][6] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean; stracks[i].covariance = cov
                stracks[i].kf_depth.predict()

    def activate(self, kalman_filter, frame_id):
        self.kalman_filter = kalman_filter; self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0; self.state = TrackState.Tracked
        if frame_id == 1: self.is_activated = True
        self.frame_id = frame_id; self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh))
        self.kf_depth.update(np.array([new_track.depth]))
        self.depth = self.kf_depth.x[0]; self.tracklet_len = 0
        self.state = TrackState.Tracked; self.is_activated = True; self.frame_id = frame_id
        if new_id: self.track_id = self.next_id()
        self.score = new_track.score; self.cls = new_track.cls

    def update(self, new_track, frame_id):
        self.frame_id = frame_id; self.tracklet_len += 1; new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.kf_depth.update(np.array([new_track.depth]))
        self.depth = self.kf_depth.x[0]; self.state = TrackState.Tracked
        self.is_activated = True; self.score = new_track.score
        
    @property
    def tlwh(self):
        if self.mean is None: return self._tlwh.copy()
        ret = self.mean[:4].copy(); ret[2] *= ret[3]; ret[:2] -= ret[2:] / 2
        return ret
        
    @property
    def tlbr(self):
        ret = self.tlwh.copy(); ret[2:] += ret[:2]
        return ret
        
    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy(); ret[:2] += ret[2:] / 2
        if ret[3] > 0: ret[2] /= ret[3]
        return ret
        
    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy(); ret[2:] -= ret[:2]
        return ret
        
    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy(); ret[2:] += ret[:2]
        return ret
        
    def __repr__(self):
        return f'OT_{self.track_id}_({self.start_frame}-{self.end_frame})'


class ByteTracker:
    # [修改] 构造函数现在只接收 'args'
    def __init__(self, args, frame_rate):
        self.tracked_stracks = []; self.lost_stracks = []; self.removed_stracks = []
        self.frame_id = 0
        self.args = args
        
        # [修改] 所有参数都从 args 中读取
        self.track_high_thresh = args.track_high_thresh
        self.track_low_thresh = args.track_low_thresh
        self.new_track_thresh = args.new_track_thresh
        
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilterXYAH()
        
        # [修改] 从 args 中读取 maha 阈值
        self.maha_thresh = args.maha_thresh
        self.motion_maha_thresh = args.motion_maha_thresh

    def update(self, results, img=None):
        self.frame_id += 1
        activated_stracks = []; refind_stracks = []; lost_stracks = []; removed_stracks = []
        
        scores=results[:, 4]; bboxes=results[:, :4]; depths=results[:, 6]; classes=results[:, 5]
        remain_inds = scores > self.track_high_thresh
        inds_low = np.logical_and(scores > self.track_low_thresh, scores < self.track_high_thresh)
        
        dets=bboxes[remain_inds]; dets_low=bboxes[inds_low]; scores_high=scores[remain_inds]
        scores_low=scores[inds_low]; classes_high=classes[remain_inds]; classes_low=classes[inds_low]
        depths_high=depths[remain_inds]; depths_low=depths[inds_low]
        
        # [修改] 创建 STrack 实例时传递 KF 参数
        detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c, d,
                             kf_R=self.args.depth_kf_R, 
                             kf_Q_pos=self.args.depth_kf_Q_pos, 
                             kf_Q_vel=self.args.depth_kf_Q_vel)
                      for (tlbr, s, c, d) in zip(dets, scores_high, classes_high, depths_high)] if len(dets) > 0 else []
        
        unconfirmed = []; tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated: unconfirmed.append(track)
            else: tracked_stracks.append(track)
            
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20: dists = matching.fuse_score(dists, detections)
        matches, u_track_idx, u_detection_idx = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        
        for itracked, idet in matches:
            track=strack_pool[itracked]; det=detections[idet]
            if track.state == TrackState.Tracked: track.update(det, self.frame_id); activated_stracks.append(track)
            else: track.re_activate(det, self.frame_id, new_id=False); refind_stracks.append(track)
            
        unmatched_tracks_after_iou = [strack_pool[i] for i in u_track_idx]
        unmatched_detections_after_iou = [detections[i] for i in u_detection_idx]
        
        if len(unmatched_tracks_after_iou) > 0 and len(unmatched_detections_after_iou) > 0:
            motion_dists = motion_mahalanobis_distance(unmatched_tracks_after_iou, unmatched_detections_after_iou)
            maha_dists = mahalanobis_distance_depth(unmatched_tracks_after_iou, unmatched_detections_after_iou)
            
            for i, track in enumerate(unmatched_tracks_after_iou):
                innovation_cov = track.kf_depth.P[0, 0] + track.kf_depth.R[0, 0]
                # [修改] 使用来自 args 的 depth_gate_factor
                depth_gate_threshold = self.args.depth_gate_factor * np.sqrt(innovation_cov)
                predicted_depth = track.kf_depth.x[0]

                for j, det in enumerate(unmatched_detections_after_iou):
                    if motion_dists[i, j] > self.motion_maha_thresh:
                        maha_dists[i, j] = np.inf
                        continue
                    if abs(predicted_depth - det.depth) > depth_gate_threshold:
                        maha_dists[i, j] = np.inf
            
            # [修改] 此处使用 self.maha_thresh (在 __init__ 中从 args 设置)
            matches_2, u_track_idx_2, u_detection_idx_2 = matching.linear_assignment(maha_dists, thresh=self.maha_thresh)
            
            for itracked, idet in matches_2:
                track=unmatched_tracks_after_iou[itracked]; det=unmatched_detections_after_iou[idet]
                if track.state == TrackState.Tracked: track.update(det, self.frame_id); activated_stracks.append(track)
                else: track.re_activate(det, self.frame_id, new_id=False); refind_stracks.append(track)
            
            r_tracked_stracks = [unmatched_tracks_after_iou[i] for i in u_track_idx_2 if unmatched_tracks_after_iou[i].state == TrackState.Tracked]
            detections_after_depth = [unmatched_detections_after_iou[i] for i in u_detection_idx_2]
        else:
            r_tracked_stracks = [t for t in unmatched_tracks_after_iou if t.state == TrackState.Tracked]
            detections_after_depth = unmatched_detections_after_iou
            
        # [修改] 创建 STrack 实例时传递 KF 参数
        detections_low = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c, d,
                                 kf_R=self.args.depth_kf_R, 
                                 kf_Q_pos=self.args.depth_kf_Q_pos, 
                                 kf_Q_vel=self.args.depth_kf_Q_vel)
                          for (tlbr, s, c, d) in zip(dets_low, scores_low, classes_low, depths_low)]
        
        dists = matching.iou_distance(r_tracked_stracks, detections_low)
        # [修改] 使用来自 args 的 second_match_thresh
        matches, u_track_low_idx, u_detection_low_idx = matching.linear_assignment(dists, thresh=self.args.second_match_thresh)
        
        for itracked, idet in matches:
            track=r_tracked_stracks[itracked]; det=detections_low[idet]
            if track.state == TrackState.Tracked: track.update(det, self.frame_id); activated_stracks.append(track)
            else: track.re_activate(det, self.frame_id, new_id=False); refind_stracks.append(track)
            
        for it in u_track_low_idx:
            track=r_tracked_stracks[it]
            if not track.state == TrackState.Lost: track.mark_lost(); lost_stracks.append(track)
            
        dists = matching.iou_distance(unconfirmed, detections_after_depth)
        if not self.args.mot20: dists = matching.fuse_score(dists, detections_after_depth)
        # [修改] 使用来自 args 的 third_match_thresh
        matches, u_unconfirmed_idx, u_detection_final_idx = matching.linear_assignment(dists, thresh=self.args.third_match_thresh)
        
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections_after_depth[idet], self.frame_id); activated_stracks.append(unconfirmed[itracked])
            
        for it in u_unconfirmed_idx:
            track=unconfirmed[it]; track.mark_removed(); removed_stracks.append(track)
            
        for inew in u_detection_final_idx:
            track = detections_after_depth[inew]
            if track.score > self.new_track_thresh: track.activate(self.kalman_filter, self.frame_id); activated_stracks.append(track)
            
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost: track.mark_removed(); removed_stracks.append(track)
            
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        outputs = []
        for track in output_stracks:
            tlwh=track.tlwh; xyxy=STrack.tlwh_to_tlbr(tlwh); track_id=track.track_id
            score=track.score; cls=track.cls; depth=track.depth
            outputs.append(np.array([xyxy[0], xyxy[1], xyxy[2], xyxy[3], track_id, score, cls, depth]))
            
        if len(outputs) > 0: return np.stack(outputs)
        else: return np.empty((0, 8))

# ... (joint_stracks, sub_stracks, remove_duplicate_stracks 保持不变) ...
def joint_stracks(tlista, tlistb):
    exists = {}; res = []
    for t in tlista: exists[t.track_id] = 1; res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0): exists[tid] = 1; res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {}; 
    for t in tlista: stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0): del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq: dupb.append(q)
        else: dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb