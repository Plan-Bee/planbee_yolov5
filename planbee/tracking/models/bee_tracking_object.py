import math

from tracking.models import bee_movement
from tracking.models.hive_position import HivePosition
from tracking.models.bee_movement import BeeMovement
import json


class BeeTrackingObject:
	object_id: int
	start_frame_id: int
	end_frame_id: int
	end_age: int
	estimates: [(int, int)]
	angle: int  # 0° is if the bee flies "to the right on the x-axis". Angle turns clockwise
	flight_distance: float
	flies_out_of_frame: bool
	bee_movement: BeeMovement

	def __init__(self, object_id: int, start_frame_id: int, age: int, initial_estimate: tuple):
		self.object_id = object_id
		self.start_frame_id = start_frame_id
		self.end_frame_id = start_frame_id
		self.end_age = age
		self.estimates = [initial_estimate]

	def _calculate_directions(self, hive_position: HivePosition, moving_offset: int):
		"""
		Calculates where the bee is going to

		Args:
			hive_position: The position of the hive in the frame
			moving_offset: In pixels on the frame. Has to be calculated based on the image resolution!
		"""
		# Check if the bee moves more than the required offset
		if self.flight_distance < moving_offset:
			self.bee_movement = BeeMovement.NO_MOVEMENT
			return

		# Calculate angles
		greater_than_angle = (hive_position.value - 90) % 360
		smaller_than_angle = (hive_position.value + 90) % 360

		start_coordinates = self.estimates[0]
		end_coordinates = self.estimates[-1]

		x_difference = end_coordinates[0] - start_coordinates[0]
		y_difference = end_coordinates[1] - start_coordinates[1]

		self.angle = int(math.degrees(math.atan2(y_difference, x_difference)) % 360)

		if self.flies_out_of_frame and self.angle > greater_than_angle or self.angle < smaller_than_angle:
			self.bee_movement = BeeMovement.TO_HIVE
		elif self.flies_out_of_frame:
			self.bee_movement = BeeMovement.FROM_HIVE
		else:
			self.bee_movement = BeeMovement.NO_MOVEMENT

	# print(f'{self.object_id}: {self.angle} ({x_difference}, {y_difference}), to: {flies_to}')

	def _calculate_distances(self):
		start_coordinates = self.estimates[0]
		end_coordinates = self.estimates[-1]

		x_difference = end_coordinates[0] - start_coordinates[0]
		y_difference = end_coordinates[1] - start_coordinates[1]

		hypo = math.sqrt(x_difference ** 2 + y_difference ** 2)

		self.flight_distance = hypo

	def determine_movement(self, hive_position, moving_offset):
		self._calculate_distances()
		self._calculate_directions(hive_position, moving_offset)

	def save_to_json(self, path):
		model_as_dict = self.__dict__

		with open(path, 'w', encoding='utf-8') as file:
			json.dump(model_as_dict, file, ensure_ascii=False, indent=4)

			file.close()
