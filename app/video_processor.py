"""
ClaimLens — Video Processor
Extracts distinct, high-quality frames from uploaded claim videos.
Performs continuity check to detect cut/edited videos (fraud signal).
"""
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFrame:
    frame_index: int        # original frame number in video
    timestamp_sec: float    # time in video when frame was taken
    image: np.ndarray       # BGR image array (H x W x 3)
    blur_score: float       # higher = sharper (Laplacian variance)
    is_usable: bool         # passed blur + exposure checks


@dataclass
class VideoAnalysis:
    frames: List[ExtractedFrame]
    total_frames: int
    duration_sec: float
    fps: float
    continuity_score: float     # 1.0 = fully continuous, <0.5 = likely edited
    has_cuts: bool              # True = jump detected → fraud flag
    cut_timestamps: List[float] # timestamps where cuts were detected


class VideoProcessor:
    """
    CHANGE THESE THRESHOLDS based on your testing:
      - BLUR_THRESHOLD: raise if too many blurry frames accepted
      - SCENE_DIFF_THRESHOLD: lower if cut detection missing edits
      - FRAMES_PER_SECOND_SAMPLE: raise for more frames (slower)
    """

    BLUR_THRESHOLD         = 80.0    # Laplacian variance below this = blurry
    SCENE_DIFF_THRESHOLD   = 45.0    # Histogram diff above this = scene cut
    FRAMES_PER_SECOND_SAMPLE = 1.5   # sample 1.5 frames per second of video
    MIN_USABLE_FRAMES      = 5       # reject video if fewer usable frames
    IMG_SIZE               = (300, 300)  # resize for model input

    def process_video(self, video_path: str) -> VideoAnalysis:
        """Main entry point — process a video file end to end."""
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        try:
            fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = total_frames / fps

            # ── Step 1: Sample frames across the video
            sampled = self._sample_frames(cap, fps, total_frames)

            # ── Step 2: Quality filter each frame
            extracted = [self._assess_frame(img, idx, idx/fps)
                         for idx, img in sampled]

            # ── Step 3: Cut detection (fraud check)
            continuity, has_cuts, cut_times = self._detect_cuts(sampled)

            usable = [f for f in extracted if f.is_usable]
            logger.info(
                f"Video processed: {len(usable)}/{len(extracted)} usable frames, "
                f"cuts={'YES' if has_cuts else 'no'}"
            )

            return VideoAnalysis(
                frames=extracted,
                total_frames=total_frames,
                duration_sec=duration_sec,
                fps=fps,
                continuity_score=continuity,
                has_cuts=has_cuts,
                cut_timestamps=cut_times,
            )
        finally:
            cap.release()

    # ── Private helpers ────────────────────────────────────────────────────

    def _sample_frames(
        self, cap: cv2.VideoCapture, fps: float, total: int
    ) -> List[Tuple[int, np.ndarray]]:
        """Sample frames at FRAMES_PER_SECOND_SAMPLE rate."""
        step = max(1, int(fps / self.FRAMES_PER_SECOND_SAMPLE))
        frames = []
        idx = 0
        while idx < total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append((idx, frame))
            idx += step
        return frames

    def _assess_frame(
        self, image: np.ndarray, frame_idx: int, ts: float
    ) -> ExtractedFrame:
        """Compute blur score and exposure check for one frame."""
        gray       = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Simple exposure check — avoid pure black or pure white frames
        mean_brightness = gray.mean()
        exposure_ok = 20 < mean_brightness < 235

        is_usable = blur_score >= self.BLUR_THRESHOLD and exposure_ok

        # Resize for model input
        resized = cv2.resize(image, self.IMG_SIZE)

        return ExtractedFrame(
            frame_index=frame_idx,
            timestamp_sec=ts,
            image=resized,
            blur_score=float(blur_score),
            is_usable=is_usable,
        )

    def _detect_cuts(
        self, frames: List[Tuple[int, np.ndarray]]
    ) -> Tuple[float, bool, List[float]]:
        """
        Detect abrupt scene changes that indicate the video was edited.
        Uses colour histogram difference between consecutive frames.

        CHANGE: SCENE_DIFF_THRESHOLD if you get too many false positives
        on videos recorded in changing light conditions (e.g. driving
        under trees). Raise it to ~55–60 in that case.
        """
        if len(frames) < 2:
            return 1.0, False, []

        diffs = []
        cut_times = []

        for i in range(1, len(frames)):
            prev_idx, prev_img = frames[i-1]
            curr_idx, curr_img = frames[i]

            # Compute normalised histogram for each frame
            h1 = self._hist(prev_img)
            h2 = self._hist(curr_img)
            diff = cv2.compareHist(h1, h2, cv2.HISTCMP_CHISQR)
            diffs.append(diff)

            if diff > self.SCENE_DIFF_THRESHOLD:
                cut_times.append(curr_idx / 30.0)   # approx timestamp

        has_cuts = len(cut_times) > 0
        # Continuity = fraction of transitions that were smooth
        n_smooth  = sum(1 for d in diffs if d <= self.SCENE_DIFF_THRESHOLD)
        continuity = n_smooth / len(diffs) if diffs else 1.0

        return continuity, has_cuts, cut_times

    def _hist(self, img: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h   = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(h, h)
        return h

    def get_usable_images_as_arrays(
        self, analysis: VideoAnalysis
    ) -> List[np.ndarray]:
        """Return resized RGB float32 arrays ready for model inference."""
        out = []
        for f in analysis.frames:
            if f.is_usable:
                rgb = cv2.cvtColor(f.image, cv2.COLOR_BGR2RGB)
                arr = rgb.astype(np.float32) / 255.0
                out.append(arr)
        return out
