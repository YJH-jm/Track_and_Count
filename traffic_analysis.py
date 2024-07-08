
import argparse
import cv2

from vid_process import VideoProcessor
from detect import Detector

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Traffic Flow Analysis with Inference and ByteTrack"
    )

    # parser.add_argument(
    #     "--model_id",
    #     default="vehicle-count-in-drone-video/6",
    #     help="Roboflow model ID",
    #     type=str,
    # )
    # parser.add_argument(
    #     "--roboflow_api_key",
    #     default=None,
    #     help="Roboflow API KEY",
    #     type=str,
    # )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--yolo_weights",
        default="vehicle-count-in-drone-video/6",
        help="Roboflow model ID",
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

    return args

def main(args):
    vid = VideoProcessor(args.source_video_path, args.confidence_threshold, args.iou_threshold)
    detector = Detector(args.yolo_weights, args.confidence_threshold, args.iou_threshold) 

    frames = vid.get_frames()

    for frame in frames:
        result = detector.detection(frame)
        annot_frame = vid.annotate_frame(frame, result)
        cv2.imshow("test", annot_frame)
        
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
if __name__ == "__main__":

    args = parse_arguments()
    main(args)



