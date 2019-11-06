import sys
import grpc
import requests

# import the generated classes
import services.service_spec.segmentation_pb2_grpc as grpc_bt_grpc
import services.service_spec.segmentation_pb2 as grpc_bt_pb2

from services import registry

TEST_URL = "https://raw.githubusercontent.com/singnet/dnn-model-services/master/docs/assets/users_guide/bulldog.jpg"

if __name__ == "__main__":

    try:
        test_flag = False
        if len(sys.argv) == 2:
            if sys.argv[1] == "auto":
                test_flag = True

        endpoint = input("Endpoint (localhost:{}): ".format(registry["mask_rcnn_server"]["grpc"])) if not test_flag else ""
        if endpoint == "":
            endpoint = "localhost:{}".format(registry["mask_rcnn_server"]["grpc"])

        grpc_method = input("Method (segment): ") if not test_flag else "segment"
        img_path = input("Image (URL): ") if not test_flag else TEST_URL

        # open a gRPC channel
        channel = grpc.insecure_channel("{}".format(endpoint))

        img_data = None
        if "http://" in img_path or "https://" in img_path:
            header = {'User-Agent': 'Mozilla/5.0 (Windows NT x.y; Win64; x64; rv:9.0) Gecko/20100101 Firefox/10.0'}
            r = requests.get(img_path, headers=header, allow_redirects=True)
            img_data = r.content

        if img_data is None:
            print("Invalid Image URL!")
            exit(1)

        img = grpc_bt_pb2.Image(content=img_data)
        request = grpc_bt_pb2.Request(img=img)

        stub = grpc_bt_grpc.SemanticSegmentationStub(channel)

        if grpc_method == "segment":
            response = stub.segment(request)
            print("Classes detected:", response.class_names)
            if not response.class_names:
                exit(1)
        else:
            print("Invalid method!")
            exit(1)

    except Exception as e:
        print(e)
        exit(1)
