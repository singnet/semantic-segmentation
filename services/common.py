import time
import logging
import io
import os
import sys
import skimage
import warnings
import base64

import skimage.io
from skimage import img_as_uint
import matplotlib.pyplot as plt
import PIL, PIL.Image

from multiprocessing import Pool

logging.basicConfig(level=10, format="%(asctime)s - [%(levelname)8s] - %(name)s - %(message)s")
log = logging.getLogger(__package__ + "." + __name__)

# COCO Class names
# Index of the class in the list is its ID.
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")

def load_model():
    import mrcnn.model as modellib
    from mrcnn.config import Config

    class InferenceConfig(Config):
        """Configuration for training on MS COCO.
        Derives from the base Config class and overrides values specific
        to the COCO dataset.
        """
        # Give the configuration a recognizable name
        NAME = "coco"

        # We use a GPU with 12GB memory, which can fit two images.
        # Adjust down if you use a smaller GPU.
        #IMAGES_PER_GPU = 2
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        IMAGES_PER_GPU = 1

        # Uncomment to train on 8 GPUs (default is 1)
        # GPU_COUNT = 8
        GPU_COUNT = 1

        # Number of classes (including background)
        NUM_CLASSES = 1 + 80  # COCO has 80 classes

    config = InferenceConfig()

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "models", "mask_rcnn_coco.h5")
    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    log.info("Mask_RCNN weights loaded and model initialised")
    return model

def init():
    sys.path.append(os.path.join(ROOT_DIR, "mask_rcnn"))  # To find local version of the library
    # Import COCO config
    sys.path.append(os.path.join(ROOT_DIR, "mask_rcnn/samples/coco/"))  # To find local version

    #config.display()


def fig2png_buffer(fig):
    log.info("fig2png")
    fig.canvas.draw()

    buffer = io.BytesIO()

    pilImage = PIL.Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    print("Read canvas to pil image")
    pilImage.save(buffer, "PNG")
    
    return buffer


def segment_image(img, visualize=False):
    import tensorflow as tf
    from keras import backend as K
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    K.set_session(sess)

    init()

    model = load_model()

    # Run detection
    results = model.detect([img], verbose=1)
    r = results[0]

    print("Model inference completed")

    # We need to copy/serialise some of this data otherwise
    # it will be lost when this spawned process finishes
    if visualize:
        print("Visualisation requested")
        from mrcnn import visualize
        # Visualize results
        fig, ax = plt.subplots(1, figsize=plt.figaspect(img))
        ax.set_axis_off()
        fig.subplots_adjust(0, 0, 1, 1)

        visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'], ax=ax)
        print("Saving visualise result")
        viz_img_buff = fig2png_buffer(fig)

        r["resultImage"] = viz_img_buff.getvalue()

    r['rois'] = r['rois'].tolist()
    r['class_ids'] = r['class_ids'].tolist()
    r['class_names'] = [class_names[i] for i in r['class_ids']]
    r['known_classes'] = class_names
    r['scores'] = r['scores'].tolist()
    masks = r['masks']
    r['masks'] = []
    for i in range(masks.shape[2]):
        # convert mask arrays into gray-scale pngs, then base64 encode them
        buff = io.BytesIO()
        # skimage annoying spams warnings when the mask is below a certain pixel area
        # proportional to the image size.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skimage.io.imsave(buff, img_as_uint(masks[:, :, i]))
        r['masks'].append(buff.getvalue())
    
    del model
    sess.close()
    return r
