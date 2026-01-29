import os
import streamlit as st
import cv2
import numpy as np
import tempfile
from typing import List, Tuple
from ultralytics import YOLO
from deepsort_tracker import DeepSORT, TrackVisualizer


class SpaceDebrisSystem:
    def __init__(self, model_name: str = 'runs/detect/train/weights/best.pt', device: str = '0'):
        self.detector = YOLO(model_name)
        try:
            self.detector.to(device)
        except Exception:
            pass
        self.tracker = DeepSORT(
            max_age=100,
            min_hits=15,
            iou_threshold=0.15,
            feature_dim=128,
            device=device if device != '0' else 'cuda'
        )
        self.device = device

    def detect(self, frame: np.ndarray, conf: float = 0.45) -> List[List[float]]:
        if frame is None or frame.size == 0:
            return []
        if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_bgr = frame
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.detector.predict(frame_rgb, conf=conf, verbose=False)[0]
        detections: List[List[float]] = []
        if getattr(results, "boxes", None):
            boxes = results.boxes.xyxy.cpu().numpy() if hasattr(results.boxes, 'xyxy') else np.array([])
            confs = results.boxes.conf.cpu().numpy() if hasattr(results.boxes, 'conf') else np.array([])
            clss = results.boxes.cls.cpu().numpy() if hasattr(results.boxes, 'cls') else np.array([])
            for idx in range(len(boxes)):
                box = boxes[idx]
                confidence = float(confs[idx]) if idx < len(confs) else 1.0
                class_id = int(clss[idx]) if len(clss) > 0 and idx < len(clss) else 0
                detections.append([float(box[0]), float(box[1]), float(box[2]), float(box[3]), confidence, class_id])
        return detections

    def process_frame(self, frame: np.ndarray, frame_idx: int, conf: float = 0.45) -> Tuple[List, List, np.ndarray]:
        if frame is None:
            return [], [], np.zeros((1, 1, 3), dtype=np.uint8)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        detections = self.detect(frame, conf=conf)
        tracks = self.tracker.update(detections, frame)
        annotated = TrackVisualizer.draw_tracks(frame.copy(), tracks, show_trajectory=True, show_velocity=True)
        return detections, tracks, annotated

    def process_video(self, video_path: str, output_path: str, status_text=None, conf: float = 0.45):
        import av
        from io import BytesIO
        import numpy as np
        import cv2
        import os

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        import math

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps is None or fps <= 0 or math.isnan(fps) or math.isinf(fps):
            fps = 25

        fps = int(round(fps))
        if fps <= 0:
            fps = 25

        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to read first frame from video")

        height, width = first_frame.shape[:2]

        buffer = BytesIO()
        container = av.open(buffer, mode="w", format="mp4")

        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"

        self.tracker.reset()
        frames_processed = 0
        all_results = []

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections, tracks, annotated = self.process_frame(frame, frames_processed, conf=conf)

            if annotated is None:
                annotated = np.zeros((height, width, 3), dtype=np.uint8)

            if annotated.dtype != np.uint8:
                annotated = np.clip(annotated, 0, 255).astype(np.uint8)
            if annotated.shape[:2] != (height, width):
                annotated = cv2.resize(annotated, (width, height))

            av_frame = av.VideoFrame.from_ndarray(annotated, format="bgr24")

            for packet in stream.encode(av_frame):
                container.mux(packet)

            all_results.append({
                'frame': frames_processed,
                'detections': len(detections),
                'tracks': len(tracks),
                'track_info': tracks
            })

            frames_processed += 1

            if status_text is not None:
                status_text.text(f"Frame {frames_processed}/{total_frames}")

        for packet in stream.encode():
            container.mux(packet)

        container.close()
        cap.release()

        with open(output_path, "wb") as f:
            f.write(buffer.getvalue())

        return all_results, output_path


def main():
    st.set_page_config(page_title='Space Debris Detection & Tracking', layout='wide', initial_sidebar_state='expanded')
    st.title('Space Debris Detection & Tracking System')
    st.markdown('Powered by YOLOv8 + DeepSORT')
    with st.sidebar:
        st.header('Configuration')
        mode = st.radio('Select Mode', ['Image Detection', 'Video Tracking'])
        conf_threshold = st.slider('Confidence Threshold', min_value=0.1, max_value=0.9, value=0.45, step=0.05)
        device = st.selectbox('Device', ['0 (GPU)', 'cpu'])
    device_id = '0' if device == '0 (GPU)' else 'cpu'
    @st.cache_resource
    def load_system():
        return SpaceDebrisSystem(device=device_id)
    system = load_system()
    if mode == 'Image Detection':
        st.header('Single Image Detection')
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_file = st.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png'])
        with col2:
            detect_btn = st.button('Detect')
        if uploaded_file and detect_btn:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            tmp.write(uploaded_file.read())
            tmp.close()
            frame = cv2.imread(tmp.name)
            detections = system.detect(frame, conf_threshold)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader('Original')
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            with col2:
                st.subheader(f'Detections ({len(detections)})')
                annotated = frame.copy()
                for x1, y1, x2, y2, conf, cls in detections:
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Debris {conf:.2f}"
                    cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            st.metric('Total Detections', len(detections))
    elif mode == 'Video Tracking':
        st.header('Video Tracking with DeepSORT')
        video_file = st.file_uploader('Upload Video', type=['mp4', 'avi', 'mov'])
        if video_file:
            col1, col2 = st.columns(2)
            with col2:
                process_btn = st.button('Process Video', key='process_video')
            if process_btn:
                tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tmp_video.write(video_file.read())
                tmp_video.close()
                status_text = st.empty()
                output_path = 'output_videos/tracked_output.mp4'
                with st.spinner('Processing video...'):
                    results, output_path = system.process_video(
                        tmp_video.name,
                        output_path=output_path,
                        status_text=status_text,
                        conf=conf_threshold
                    )
                st.success('Tracking complete!')
                with open(output_path, "rb") as f:
                    st.video(f.read())
                st.subheader('Tracking Statistics')
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric('Total Frames', len(results))
                with col2:
                    unique_ids = set()
                    for r in results:
                        for t in r['track_info']:
                            unique_ids.add(t['id'])
                    st.metric('Unique Objects', len(unique_ids))
                with col3:
                    avg_tracks = np.mean([r['tracks'] for r in results]) if len(results) > 0 else 0.0
                    st.metric('Avg Active Tracks', f"{avg_tracks:.1f}")

if __name__ == '__main__':
    main()
