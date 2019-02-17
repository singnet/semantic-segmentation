import sys
import logging
import base64
import io
import os
import grpc
import asyncio
import argparse
import os.path
import time
import warnings
import concurrent.futures
from multiprocessing import Pool

import skimage.io
from skimage import img_as_uint
import matplotlib.pyplot as plt
import PIL, PIL.Image

from services import registry
import services.service_spec.segmentation_pb2 as ss_pb
import services.service_spec.segmentation_pb2_grpc as ss_grpc 

logging.basicConfig(level=10, format="%(asctime)s - [%(levelname)8s] - %(name)s - %(message)s")
log = logging.getLogger(__package__ + "." + __name__)

class SegmentationServicer(ss_grpc.SemanticSegmentationServicer):
    def __init__(self, q):
        self.q = q
        pass

    def segment(self, request, context):
        ## Marshal input
        image = request.img.content
        mimetype = request.img.mimetype

        img_data = io.BytesIO(image)
        img = skimage.io.imread(img_data)

        # Drop alpha channel if it exists
        if img.shape[-1] == 4:
            img = img[:, :, :3]
            log.debug("Dropping alpha channel from image")

        self.q.send((img,))
        result = self.q.recv()
        if isinstance(result, Exception):
            raise result

        ## Marshal output
        pb_result = ss_pb.Result(
            segmentation_img=[ss_pb.Image(content=i) for i in result["masks"]],
            debug_img=ss_pb.Image(content=result["resultImage"]),
            class_ids=[class_id for class_id in result["class_ids"]],
            class_names=[n for n in result["class_names"]],
            scores=[n for n in result["scores"]]
            #known_classes=[n for n in result["known_classes"]],
        )

        return pb_result


def serve(dispatch_queue, max_workers=1, port=7777):
    assert max_workers == 1, "No support for more than one worker"
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=max_workers))
    ss_grpc.add_SemanticSegmentationServicer_to_server(SegmentationServicer(dispatch_queue), server)
    server.add_insecure_port("[::]:{}".format(port))
    return server


def main_loop(dispatch_queue, grpc_serve_function, grpc_port, grpc_args={}):
    server = None
    if grpc_serve_function is not None:
        server = grpc_serve_function(dispatch_queue, port=grpc_port, **grpc_args)
        server.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        server.stop(0)

def worker(q):
    import services.common
    while True:
        try:
            item = q.recv()
            with Pool(1) as p:
                result = p.apply(services.common.segment_image, (item[0], True))
            q.send(result)
        except Exception as e:
            q.send(e)


if __name__ == "__main__":
    script_name = __file__
    parser = argparse.ArgumentParser(prog=script_name)
    server_name = os.path.splitext(os.path.basename(script_name))[0]
    parser.add_argument("--grpc-port", help="port to bind grpc services to", default=registry[server_name]['grpc'], type=int, required=False)
    args = parser.parse_args(sys.argv[1:])

    # Need queue system and spawning grpc server in separate process because of:
    # https://github.com/grpc/grpc/issues/16001

    import multiprocessing as mp
    pipe = mp.Pipe()
    p = mp.Process(target=main_loop, args=(pipe[0], serve, args.grpc_port))
    p.start()

    w = mp.Process(target=worker, args=(pipe[1],))
    w.start()

    p.join()
    w.join()