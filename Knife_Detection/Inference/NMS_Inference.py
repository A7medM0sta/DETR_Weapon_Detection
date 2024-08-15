import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
import os
import cv2
import supervision as sv


class ObjectDetectionWithNMS:
    """
    A class for object detection using the DETR (DEtection TRansformers) model with Non-Maximum Suppression (NMS).

    Attributes
    ----------
    image_path : str
        Path to the image file for detection.
    device : torch.device
        The device (CPU or CUDA) on which the model will run.
    checkpoint : str
        The model checkpoint to be used for DETR.
    confidence_threshold : float
        The confidence threshold for object detection.
    iou_threshold : float
        The Intersection Over Union (IOU) threshold for Non-Maximum Suppression.
    image_processor : DetrImageProcessor
        The image processor used to prepare inputs for the DETR model.
    model : DetrForObjectDetection
        The DETR model used for object detection.
    """

    def __init__(self, image_path, checkpoint='facebook/detr-resnet-50', confidence_threshold=0.5, iou_threshold=0.8):
        """
        Initializes the ObjectDetectionWithNMS class with specified settings.

        Parameters
        ----------
        image_path : str
            Path to the image file for detection.
        checkpoint : str
            The model checkpoint to be used for DETR.
        confidence_threshold : float
            The confidence threshold for object detection.
        iou_threshold : float
            The Intersection Over Union (IOU) threshold for Non-Maximum Suppression.
        """
        self.image_path = image_path
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.checkpoint = checkpoint
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_processor = DetrImageProcessor.from_pretrained(self.checkpoint)
        self.model = DetrForObjectDetection.from_pretrained(self.checkpoint)
        self.model.to(self.device)

    def load_image(self):
        """
        Loads the image from the specified path.

        Returns
        -------
        image : ndarray
            The loaded image.
        """
        image = cv2.imread(self.image_path)
        return image

    def predict(self, image):
        """
        Performs object detection on the provided image and applies NMS.

        Parameters
        ----------
        image : ndarray
            The input image for detection.

        Returns
        -------
        detections : Detections
            The detection results including bounding boxes, labels, and scores, post-NMS.
        """
        with torch.no_grad():
            inputs = self.image_processor(images=image, return_tensors='pt').to(self.device)
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([image.shape[:2]]).to(self.device)
            results = self.image_processor.post_process_object_detection(
                outputs=outputs,
                threshold=self.confidence_threshold,
                target_sizes=target_sizes
            )[0]

        detections = sv.Detections.from_transformers(transformers_results=results).with_nms(
            threshold=self.iou_threshold)
        return detections

    def annotate_image(self, image, detections):
        """
        Annotates the image with detection results.

        Parameters
        ----------
        image : ndarray
            The input image to be annotated.
        detections : Detections
            The detection results including bounding boxes, labels, and scores.

        Returns
        -------
        frame : ndarray
            The annotated image.
        """
        labels = [
            f"{self.model.config.id2label[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
        box_annotator = sv.BoxAnnotator()
        frame = box_annotator.annotate(scene=image, detections=detections, labels=labels)
        return frame

    def show_image(self, frame, size=(16, 16)):
        """
        Displays the annotated image.

        Parameters
        ----------
        frame : ndarray
            The annotated image to be displayed.
        size : tuple
            The size of the displayed image in inches.
        """
        sv.show_frame_in_notebook(frame, size)


# Example Usage
def main():
    image_path = "/content/Dangerous-Stuff-2/train/20221013_212559_jpg.rf.5051078aeb7a69734cb23287e1d2d65b.jpg"
    detector = ObjectDetectionWithNMS(image_path)
    image = detector.load_image()
    detections = detector.predict(image)
    annotated_image = detector.annotate_image(image, detections)
    detector.show_image(annotated_image)

if __name__ == "__main__":
    main()