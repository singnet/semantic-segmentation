# Semantic Segmentation

This repository contains a SingularityNET service to do semantic segmentation on images.

Currently it just supports the Mask_RCNN approach, pulling in Matterport's Mask_RCNN implementation.

## TODO

- [ ] Support other semantic segmentation algorithms/models.
- [ ] Publish information about the semantic classes each service knows about using
  the metadata URI in the SingularityNET Agent contract. 
- [ ] grpc model