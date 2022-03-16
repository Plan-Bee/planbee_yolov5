import numpy as np
import torch
import yolov5
from typing import Union, List, Optional


class YOLO:
	def __init__(self, model_path: str, device: Optional[str] = None):
		if device is not None and "cuda" in device and not torch.cuda.is_available():
			raise Exception("Selected device='cuda', but cuda is not available to Pytorch.")
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
