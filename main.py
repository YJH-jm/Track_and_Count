import argparse
from tqdm import tqdm
import os
import sys
from typing import List
from onemetric.cv.utils.iou import box_iou_batch
from yolox.tracker.byte_tracker import BYTETracker, STrack
sys.path.append(f"{os.getcwd()}/ByteTrack")
import numpy as np

import cv2

from ultralytics import YOLO

from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point

from tracker import get_tracker_model

LINE_START = Point(50, 1500)
LINE_END = Point(3840-50, 1500)



# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections,
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids

def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("vid_path", type=str, help="img file or folder path")

    return parser.parse_args()


def main(args):

    model_path = "./saved/yolov8x.pt"
    model = YOLO(model_path)
    model.fuse()

    SAVED_VID_PATH = f"results/{args.vid_path.split('/')[-1] if args.vid_path in '/' else args.vid_path}"

    
    generator = get_video_frames_generator(args.vid_path)
    
    line_counter = LineCounter(start=LINE_START, end=LINE_END)

    byte_tracker = get_tracker_model()
    vid_info = VideoInfo.from_video_path(args.vid_path)

    box_annot = BoxAnnotator(color=ColorPalette(), thickness=4, text_thickness=4, text_scale=2)
    line_annot = LineCounterAnnotator(thickness=4, text_thickness=4, text_scale=2)
    
    with VideoSink(SAVED_VID_PATH, vid_info) as sink:
        for frame in tqdm(generator, total=vid_info.total_frames):

            results = model(frame)[0]

            detections = Detections(
                xyxy=results.boxes.xyxy.cpu().numpy(),
                confidence=results.boxes.conf.cpu().numpy(),
                class_id=results.boxes.cls.cpu().numpy().astype('int')
            )

            tracks = byte_tracker.update(
                output_results=detections2boxes(detections=detections),
                img_info=frame.shape,
                img_size=frame.shape
            )

            tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
            detections.tracker_id = np.array(tracker_id)
            labels = [f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}" for _, confidence, class_id, tracker_id in detections]
            frame = box_annot.annotate(frame=frame, detections=detections, labels=labels)
            line_annot.annotate(frame=frame, line_counter=line_counter)
            sink.write_frame(frame)
            # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            # cv2.imshow('frame', frame)
            # if cv2.waitKey() ==27:
            #     cv2.destroyAllWindows()

if __name__ == "__main__":

    args = parse_arguments()
    main(args)