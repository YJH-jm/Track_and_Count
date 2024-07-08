import argparse
import numpy as np

import supervision as sv

COLORS = sv.ColorPalette.default()

class VideoProcessor:
    def __init__(self,
                 source_vid_path:str,
                 confidence_threshold: float=0.3,
                 iou_threshold: float=0.7) -> None:
        
        self.source_vid_path = source_vid_path
        self.target_vid_path = f"results/{source_vid_path.split('/')[-1] if source_vid_path in '/' else source_vid_path}"
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        self.box_annotator = sv.BoxAnnotator(color=COLORS)
    
    def get_frames(self):
        frame_generator = sv.get_video_frames_generator(source_path=self.source_vid_path)
        
        return frame_generator

    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        annotate_frame = frame.copy()
        annotate_frame = self.box_annotator.annotate(
            scene=annotate_frame, detections = detections,
        )

        return annotate_frame
     
    def process_frame(self):
        pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Traffic Flow Analysis with YOLO and ByteTrack"
    )

    parser.add_argument(
        "--source_weights_path",
        required=True,
        help="Path to the source weights file",
        type=str,
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        default=None,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )

    args = parser.parse_args()
    processor = VideoProcessor(
        source_weights_path=args.source_weights_path,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
    )
    processor.process_video()