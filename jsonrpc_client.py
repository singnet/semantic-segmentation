import jsonrpcclient
import sys
import os
import argparse
import base64

from services.snet import snet_setup
from services import registry


SERVER_NAME = 'mask_rcnn_server'


def main():
    script_name = sys.argv[0]
    parser = argparse.ArgumentParser(prog=script_name)

    default_endpoint = "http://127.0.0.1:{}".format(registry[SERVER_NAME]['jsonrpc'])
    parser.add_argument("--endpoint", help="jsonrpc server to connect to", default=default_endpoint,
                        type=str, required=False)
    parser.add_argument("--snet", help="call services on SingularityNet - requires configured snet CLI",
                        action='store_true')
    parser.add_argument("--image", help="path to image to apply face detection on",
                        type=str, required=True)
    parser.add_argument("--out-image", help="Render bounding box on image and save",
                        type=str, required=False)
    args = parser.parse_args(sys.argv[1:])

    with open(args.image, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode('ascii')

    endpoint = args.endpoint
    params = {"image": img_base64}
    if args.snet:
        endpoint, job_address, job_signature = snet_setup(service_name="semantic_segmentation")
        params['job_address'] = job_address
        params['job_signature'] = job_signature

    response = jsonrpcclient.request(endpoint, "semantic_segmentation", **params)

    if args.out_image:
        print("Rendering bounding box and saving to {}".format(args.out_image))
        import cv2
        image = cv2.imread(args.image)
        for d in response['faces']:
            cv2.rectangle(image, (d['x'], d['y']), (d['x'] + d['w'], d['y'] + d['h']), (0, 255, 0), 2)
        cv2.imwrite(args.out_image, image)


if __name__ == '__main__':
    main()



