import sys
import logging
import base64
import io
import os

import skimage.io
from skimage import img_as_uint
import matplotlib.pyplot as plt
import PIL, PIL.Image

from aiohttp import web
from jsonrpcserver.aio import methods
from jsonrpcserver.exceptions import InvalidParams

import services.common


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
model = None


def init():
    ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
    sys.path.append(os.path.join(ROOT_DIR, "mask_rcnn"))  # To find local version of the library

    # Import COCO config
    sys.path.append(os.path.join(ROOT_DIR, "mask_rcnn/samples/coco/"))  # To find local version
    from mrcnn import utils

    import mrcnn.model as modellib
    import coco

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "models", "mask_rcnn_coco.h5")

    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    global model
    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    print("Mask_RCNN weights loaded and model initialised")


def fig2png_buffer(fig):
    fig.canvas.draw()

    buffer = io.BytesIO()

    pilImage = PIL.Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    pilImage.save(buffer, "PNG")
    return buffer


def segment_image(img, visualize=False):
    from mrcnn import visualize

    # Run detection
    results = model.detect([img], verbose=1)
    r = results[0]
    if visualize:
        # Visualize results
        fig, ax = plt.subplots(1, figsize=plt.figaspect(img))
        ax.set_axis_off()
        fig.subplots_adjust(0, 0, 1, 1)

        visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'], ax=ax)
        viz_img_buff = fig2png_buffer(fig)

        r["resultImage"] = base64.b64encode(viz_img_buff.getvalue()).decode('ascii')

    r['rois'] = r['rois'].tolist()
    r['class_ids'] = r['class_ids'].tolist()
    r['scores'] = r['scores'].tolist()
    masks = r['masks']
    r['masks'] = []
    for i in range(masks.shape[2]):
        # convert mask arrays into gray-scale pngs, then base64 encode them
        buff = io.BytesIO()
        skimage.io.imsave(buff, img_as_uint(masks[:, :, i]))
        b64img = base64.b64encode(buff.getvalue()).decode('ascii')
        r['masks'].append(b64img)

    return r


@methods.add
async def ping():
    return 'pong'


@methods.add
async def semantic_segmentation(**kwargs):
    image = kwargs.get("image", None)

    if image is None:
        raise InvalidParams("image is required")

    binary_image = base64.b64decode(image)
    img_data = io.BytesIO(binary_image)
    img = skimage.io.imread(img_data)

    # Drop alpha channel if it exists
    if img.shape[-1] == 4:
        img = img[:, :, :3]
        log.debug("Dropping alpha channel from image")

    result = segment_image(img)

    return {'segmentation': result}


async def handle(request):
    request = await request.text()
    response = await methods.dispatch(request, trim_log_values=True)
    if response.is_notification:
        return web.Response()
    else:
        return web.json_response(response, status=response.http_status)


if __name__ == '__main__':
    parser = services.common.common_parser(__file__)
    args = parser.parse_args(sys.argv[1:])
    init()
    services.common.main_loop(None, None, handle, args)