

# DETR Weapon Detection

This project implements an object detection system using the DETR (DEtection TRansformers) model with Non-Maximum Suppression (NMS) to detect weapons in images.

## Dataset

The dataset used for this project includes images of dangerous objects. An example image path from the dataset is:
`/content/Dangerous-Stuff-2/train/20221013_212559_jpg.rf.5051078aeb7a69734cb23287e1d2d65b.jpg`

## Installation

To set up the project, follow these steps:

1. Clone the repository:
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Object Detection with NMS

The `ObjectDetectionWithNMS` class is used to perform object detection using the DETR model with NMS.

#### Initialization

To initialize the `ObjectDetectionWithNMS` class, provide the path to the image file for detection:
```python
detector = ObjectDetectionWithNMS(image_path)
```

#### Load Image

To load the image from the specified path:
```python
image = detector.load_image()
```

#### Predict

To perform object detection on the loaded image and apply NMS:
```python
detections = detector.predict(image)
```

#### Annotate Image

To annotate the image with detection results:
```python
annotated_image = detector.annotate_image(image, detections)
```

#### Show Image

To display the annotated image:
```python
detector.show_image(annotated_image)
```

### Example Usage

Here is an example of how to use the `ObjectDetectionWithNMS` class:
```python
def main():
    image_path = "/content/Dangerous-Stuff-2/train/20221013_212559_jpg.rf.5051078aeb7a69734cb23287e1d2d65b.jpg"
    detector = ObjectDetectionWithNMS(image_path)
    image = detector.load_image()
    detections = detector.predict(image)
    annotated_image = detector.annotate_image(image, detections)
    detector.show_image(annotated_image)

if __name__ == "__main__":
    main()
```

## Training

To train the DETR model, follow the instructions provided in the official DETR repository. Ensure you have the necessary dataset and training configurations set up before starting the training process.
```