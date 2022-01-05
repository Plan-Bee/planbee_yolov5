import copy
import math

import numpy as np
import torch
import yolov5
from typing import Union, List, Optional
from enum import Enum

import norfair
from norfair import Detection, Tracker, Video

max_distance_between_points: int = 30


class BeeTrackingObject:
	object_id: int
	start_frame_id: int
	end_frame_id: int
	end_age: int
	estimates: [(int, int)]
	angle: int  # 0° is if the bee flies "to the right on the x-axis". Angle turns clockwise
	flies_out_of_frame: bool
	flies_into_hive: bool

	def __init__(self, object_id: int, start_frame_id: int, age: int, initial_estimate: tuple):
		self.object_id = object_id
		self.start_frame_id = start_frame_id
		self.end_frame_id = start_frame_id
		self.end_age = age
		self.estimates = [initial_estimate]


class HivePosition(Enum):
	RIGHT = 0
	BOTTOM_RIGHT = 45
	BOTTOM = 90
	BOTTOM_LEFT = 135
	LEFT = 180
	TOP_LEFT = 225
	TOP = 270
	TOP_RIGHT = 315


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
		track_points: str = "centroid") -> [BeeTrackingObject]:
	model = YOLO(detector_path, device=device)

	video = Video(input_path=file, output_path=output_path)
	tracker = Tracker(
		distance_function=euclidean_distance,
		distance_threshold=max_distance_between_points,
	)

	# Save the tracked objects for each frame
	total_tracked_objects: [[]] = []

	frame_count = 0

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

		if len(total_tracked_objects) >= 100:
			frame_count = video.frame_counter
			break

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

	bee_list = converted_object_map.values()

	# Check if bees flew out of frame
	for bee in bee_list:
		bee.flies_out_of_frame = bee.end_frame_id < frame_count - 1

	return bee_list


def get_directions(tracked_bees: [BeeTrackingObject], hive_position: HivePosition) -> [BeeTrackingObject]:
	# Calculate angles
	greater_than_angle = (hive_position.value - 90) % 360
	smaller_than_angle = (hive_position.value + 90) % 360

	for bee in tracked_bees:
		start_coordinates = bee.estimates[0]
		end_coordinates = bee.estimates[-1]

		x_difference = end_coordinates[0] - start_coordinates[0]
		y_difference = end_coordinates[1] - start_coordinates[1]

		bee.angle = math.degrees(math.atan2(y_difference, x_difference)) % 360
		bee.flies_into_hive = bee.flies_out_of_frame and (
				bee.angle > greater_than_angle or bee.angle < smaller_than_angle)

		flies_to = "Hive" if bee.flies_into_hive else "Away" if bee.flies_out_of_frame else "Nowhere"

		print(
			f'{bee.object_id}: {bee.angle} ({x_difference}, {y_difference}), to: {flies_to}')

	return tracked_bees


if __name__ == '__main__':
	tracked_bees = track_bees(
		'datasets/bees/videos/2021-10-28.mp4',
		'beeyolov5/runs/track/2021-10-28.mp4'
	)
	tracked_bees = get_directions(tracked_bees, HivePosition.BOTTOM_RIGHT)
