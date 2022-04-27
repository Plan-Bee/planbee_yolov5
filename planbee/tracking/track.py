import copy
import json
import math

import cv2
import numpy as np
import torch

from typing import List

import norfair
from norfair import Detection, Tracker, Video, Paths

from planbee_models.bee_movement import BeeMovement
from planbee_models.bee_tracking_object import BeeTrackingObject
from planbee_models.hive_position import HivePosition
from planbee_models.yolo import YOLO
from datetime import datetime, timedelta, time
from planbee.persistence import mariadb_connector as db

max_distance_between_points: int = 30


def euclidean_distance(detection, tracked_object):
	return np.linalg.norm(detection.points - tracked_object.estimate)


def yolo_detections_to_norfair_detections(
		yolo_detections: torch.tensor,
		track_points: str = 'centroid'  # bbox or centroid
) -> List[Detection]:
	"""
	convert detections_as_xywh to norfair detections
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
		video: Video,
		detector_path: str = "yolov5/runs/train/original/weights/best.pt",
		img_size: int = 720,
		conf_thresh: float = 0.25,
		iou_thresh: float = 0.45,
		device: str = 'cuda',
		track_points: str = "centroid") -> [BeeTrackingObject]:
	model = YOLO(detector_path, device=device)

	tracker = Tracker(
		distance_function=euclidean_distance,
		distance_threshold=max_distance_between_points,
	)

	# Save the tracked objects for each frame
	total_tracked_objects: [[]] = []

	detections_per_frame = {}

	for frame in video:
		if video.frame_counter % 1500 == 0:
			print(f"{video.frame_counter} frames done")

		yolo_detections = model(
			frame,
			conf_threshold=conf_thresh,
			iou_threshold=iou_thresh,
			image_size=img_size
		)
		detections = yolo_detections_to_norfair_detections(yolo_detections, track_points=track_points)
		tracked_objects = tracker.update(detections=detections)

		detections_per_frame[video.frame_counter] = detections

		total_tracked_objects.append(copy.deepcopy(tracked_objects))

		if track_points == 'centroid':
			norfair.draw_points(frame, detections)
		elif track_points == 'bbox':
			norfair.draw_boxes(frame, detections)
		norfair.draw_tracked_objects(frame, tracked_objects)

		# Draw paths
		# paths = Paths(get_points_to_draw=detections, attenuation=0)
		# paths.draw(frame, tracked_objects)

		video.write(frame)

	# if len(total_tracked_objects) >= 100:
	# 	break

	frame_count = video.frame_counter

	# Convert list from list of images to list of objects
	converted_object_map = {}

	for image_index, image in enumerate(total_tracked_objects):
		for detected_object in image:
			if detected_object.id in converted_object_map:
				obj: BeeTrackingObject = converted_object_map[detected_object.id]
				obj.end_frame_id = image_index
				obj.end_age = detected_object.age
				obj.position_estimates.append((detected_object.estimate[0][0], detected_object.estimate[0][1]))
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
		bee.flies_out_of_frame = bee.end_frame_id < frame_count - video.output_fps

	return bee_list, detections_per_frame


def calculate_moving_offset(video: Video, percentage: int) -> ():
	vid_cap = cv2.VideoCapture(video.input_path)
	width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	hypo = math.sqrt(width ** 2 + height ** 2)

	return hypo * percentage / 100


def get_video_fps(video: Video) -> int:
	vid_cap = cv2.VideoCapture(video.input_path)

	return int(vid_cap.get(cv2.CAP_PROP_FPS))


def get_timestamp(frame: int, fps: float, start_time: datetime) -> float:
	delta = timedelta(seconds=frame / fps)
	return (start_time + delta).timestamp()


def save_bee_paths_to_json(bees: [BeeTrackingObject]):
	bee_dicts = []

	for this_bee in bees:
		bee_dicts.append(this_bee.get_attribute_dict())

	with open('bee_paths_slomo_15s.json', 'w', encoding='utf-8') as file:
		json.dump(bee_dicts, file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
	print(f'Start: {datetime.now()}')

	video = Video(
		input_path='datasets/bees/videos/PlanBee_Tracking_Frederick_4.mp4',
		output_path='yolov5/runs/track/PlanBee_Tracking_Frederick_4.mp4'
	)
	# video = Video(
	# 	input_path='datasets/bees/videos/PlanBee_Slomo_1080p50_5s.mp4',
	# 	output_path='yolov5/runs/track/PlanBee_Slomo_1080p50_5s.mp4'
	# )
	start_time = datetime.strptime('2022-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')  # TODO change

	print(f'Track: {datetime.now()}')

	tracked_bees, detections_per_frame = track_bees(video)

	print(f'Track done {datetime.now()}')

	moving_offset = calculate_moving_offset(video, 2)
	fps = get_video_fps(video)

	for bee in tracked_bees:
		bee.determine_movement(HivePosition.BOTTOM, moving_offset)

	#save_bee_paths_to_json(tracked_bees)

	# Now every bee has values for start_frame, end_frame and state of bee_movement
	# Furthermore we can transform the list of bees to a list form of timeseries

	bee_data: {float, list[int]} = {}

	for bee in tracked_bees:
		for frame_id in range(bee.start_frame_id, bee.end_frame_id):
			timestamp = get_timestamp(frame_id, fps, start_time)

			if timestamp not in bee_data.keys():
				bee_data[timestamp] = [0, 0, 0, 0]

			if bee.bee_movement == BeeMovement.TO_HIVE:
				bee_data[timestamp][0] += 1
			elif bee.bee_movement == BeeMovement.FROM_HIVE:
				bee_data[timestamp][1] += 1
			elif bee.bee_movement == BeeMovement.NO_MOVEMENT:
				bee_data[timestamp][2] += 1

	for frame_id, detections in detections_per_frame.items():
		timestamp = get_timestamp(frame_id, fps, start_time)

		if timestamp not in bee_data.keys():
			bee_data[timestamp] = [0, 0, 0, len(detections)]
		else:
			bee_data[timestamp][3] = len(detections)

	conn = db.get_connection()
	db.insert_tracking_data(conn, bee_data, 2, 6)

	# print(bee_data)

	print(f'Done: {datetime.now()}')
