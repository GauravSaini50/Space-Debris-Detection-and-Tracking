import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import glob

class DebrisVideoGenerator:
    def __init__(self, image_folder: str, output_folder: str = 'synthetic_videos'):
        self.image_folder = Path(image_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        self.images = self._load_images()

    def _load_images(self) -> List[np.ndarray]:
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for path in glob.glob(str(self.image_folder / ext)):
                img = cv2.imread(path)
                if img is not None:
                    images.append(img)
        return images

    def generate_video(self, num_frames: int = 150,
                       num_objects: int = 3,
                       canvas_size: Tuple[int, int] = (1280, 720),
                       output_name: str = 'debris_video.mp4',
                       motion_type: str = 'linear') -> str:
        if not self.images:
            raise ValueError("No images found in folder")
        width, height = canvas_size
        fps = 30
        output_path = self.output_folder / output_name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, canvas_size)
        objects = []
        for i in range(min(num_objects, len(self.images))):
            obj = {
                'image': cv2.resize(self.images[i % len(self.images)], (60, 60)),
                'x': np.random.randint(50, width - 50),
                'y': np.random.randint(50, height - 50),
                'vx': np.random.uniform(-8, 8),
                'vy': np.random.uniform(-8, 8),
                'rotation': np.random.uniform(0, 360),
                'rotation_speed': np.random.uniform(-5, 5)
            }
            objects.append(obj)
        for frame_idx in range(num_frames):
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            for _ in range(300):
                x, y = np.random.randint(0, width), np.random.randint(0, height)
                brightness = np.random.randint(80, 200)
                canvas[y, x] = [brightness, brightness, brightness]
            for obj in objects:
                if motion_type == 'linear':
                    obj['x'] += obj['vx']
                    obj['y'] += obj['vy']
                    if obj['x'] < -50 or obj['x'] > width + 50:
                        obj['vx'] *= -1
                    if obj['y'] < -50 or obj['y'] > height + 50:
                        obj['vy'] *= -1
                elif motion_type == 'spiral':
                    angle = frame_idx * 0.05 + np.random.random()
                    obj['x'] += obj['vx'] * np.cos(angle)
                    obj['y'] += obj['vy'] * np.sin(angle)
                obj['x'] = np.clip(obj['x'], 0, width - 60)
                obj['y'] = np.clip(obj['y'], 0, height - 60)
                obj['rotation'] += obj['rotation_speed']
                rotated = self._rotate_img(obj['image'], obj['rotation'])
                x, y = int(obj['x']), int(obj['y'])
                h, w = rotated.shape[:2]
                if 0 <= x < width and 0 <= y < height:
                    roi = canvas[max(0, y):min(height, y+h),
                                 max(0, x):min(width, x+w)]
                    obj_roi = rotated[:roi.shape[0], :roi.shape[1]]
                    canvas[max(0, y):min(height, y+h),
                          max(0, x):min(width, x+w)] = cv2.addWeighted(roi, 0.3, obj_roi, 0.7, 0)
            writer.write(canvas)
        writer.release()
        print(f"Generated: {output_path}")
        return str(output_path)

    def generate_batch(self, num_videos: int = 5, frames: int = 100) -> List[str]:
        videos = []
        for i in range(num_videos):
            motion = 'linear' if i % 2 == 0 else 'spiral'
            path = self.generate_video(
                num_frames=frames,
                num_objects= 4,
                output_name=f'debris_{i:03d}.mp4',
                motion_type=motion
            )
            videos.append(path)
        return videos

    @staticmethod
    def _rotate_img(img: np.ndarray, angle: float) -> np.ndarray:
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, matrix, (w, h))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, required=True, help='Image folder')
    parser.add_argument('--output', type=str, default='synthetic_videos')
    parser.add_argument('--videos', type=int, default=5)
    parser.add_argument('--frames', type=int, default=150)
    args = parser.parse_args()
    gen = DebrisVideoGenerator(args.images, args.output)
    gen.generate_batch(args.videos, args.frames)
