import copy

import numpy as np
import torch
import yolov5
from typing import Union, List, Optional

import norfair
from norfair import Detection, Tracker, Video

max_distance_between_points: int = 30


class BeeTrackingObject:
	object_id: int
	start_frame_id: int
	end_frame_id: int
	end_age: int
	estimates: [(int, int)]

	def __init__(self, object_id: int, start_frame_id: int, age: int, initial_estimate: tuple):
		self.object_id = object_id
		self.start_frame_id = start_frame_id
		self.end_frame_id = start_frame_id
		self.end_age = age
		self.estimates = [initial_estimate]


class YOLO:
	def __init__(self, model_path: str, device: Optional[str] = None):
		if device is not None and "cuda" in device and not torch.cuda.is_available():
			raise Exception(
				"Selected device='cuda', but cuda is not available to Pytorch."
			)
		# automatically set device if its None
		elif device is None:
			device = "cuda:0" if torch.cuda.is_available() else "cpu"
		# load model
		self.model = yolov5.load(model_path, device=device)

	def __call__(
			self,
			img: Union[str, np.ndarray],
			conf_threshold: float = 0.25,
			iou_threshold: float = 0.45,
			image_size: int = 720,
			classes: Optional[List[int]] = None
	) -> torch.tensor:

		self.model.conf = conf_threshold
		self.model.iou = iou_threshold
		if classes is not None:
			self.model.classes = classes
		detections = self.model(img, size=image_size)
		return detections


def euclidean_distance(detection, tracked_object):
	return np.linalg.norm(detection.points - tracked_object.estimate)


def yolo_detections_to_norfair_detections(
		yolo_detections: torch.tensor,
		track_points: str = 'centroid'  # bbox or centroid
) -> List[Detection]:
	"""convert detections_as_xywh to norfair detections
	"""
	norfair_detections: List[Detection] = []

	if track_points == 'centroid':
		detections_as_xywh = yolo_detections.xywh[0]
		for detection_as_xywh in detections_as_xywh:
			centroid = np.array(
				[
					detection_as_xywh[0].item(),
					detection_as_xywh[1].item()
				]
			)
			scores = np.array([detection_as_xywh[4].item()])
			norfair_detections.append(
				Detection(points=centroid, scores=scores)
			)
	elif track_points == 'bbox':
		detections_as_xyxy = yolo_detections.xyxy[0]
		for detection_as_xyxy in detections_as_xyxy:
			bbox = np.array(
				[
					[detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
					[detection_as_xyxy[2].item(), detection_as_xyxy[3].item()]
				]
			)
			scores = np.array([detection_as_xyxy[4].item(), detection_as_xyxy[4].item()])
			norfair_detections.append(
				Detection(points=bbox, scores=scores)
			)

	return norfair_detections


def track_bees(
		file: str,
		output_path: str,
		detector_path: str = "beeyolov5/bees_best.pt",
		img_size: int = 720,
		conf_thresh: float = 0.25,
		iou_thresh: float = 0.45,
		device: str = 'cuda',
		track_points: str = "centroid"):
	model = YOLO(detector_path, device=device)

	video = Video(input_path=file, output_path=output_path)
	tracker = Tracker(
		distance_function=euclidean_distance,
		distance_threshold=max_distance_between_points,
	)

	# Save the tracked objects for each frame
	total_tracked_objects: [[]] = []

	for frame in video:
		yolo_detections = model(
			frame,
			conf_threshold=conf_thresh,
			iou_threshold=iou_thresh,
			image_size=img_size
		)
		detections = yolo_detections_to_norfair_detections(yolo_detections, track_points=track_points)
		tracked_objects = tracker.update(detections=detections)

		total_tracked_objects.append(copy.deepcopy(tracked_objects))

		if track_points == 'centroid':
			norfair.draw_points(frame, detections)
		elif track_points == 'bbox':
			norfair.draw_boxes(frame, detections)
		norfair.draw_tracked_objects(frame, tracked_objects)
		video.write(frame)

	# Convert list from list of images to list of objects
	converted_object_map = {}

	for image_index, image in enumerate(total_tracked_objects):
		for detected_object in image:
			if detected_object.id in converted_object_map:
				obj: BeeTrackingObject = converted_object_map[detected_object.id]
				obj.end_frame_id = image_index
				obj.end_age = detected_object.age
				obj.estimates.append((detected_object.estimate[0][0], detected_object.estimate[0][1]))
			else:
				obj = BeeTrackingObject(
					object_id=detected_object.id,
					start_frame_id=image_index,
					age=detected_object.age,
					initial_estimate=(detected_object.estimate[0][0], detected_object.estimate[0][1])
				)
				converted_object_map[detected_object.id] = obj

	return converted_object_map


if __name__ == '__main__':
	tracked_bees = track_bees(
		'datasets/bees/videos/2021-10-28.mp4',
		'beeyolov5/runs/track/2021-10-28.mp4'
	)
