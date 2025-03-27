# CUDA_VISIBLE_DEVICES=1 torchpack dist-run -np 1 python tools/test.py configs/nuscenes/seg/fusion-bev256d2-lss.yaml pretrained/bevfusion-seg.pth --eval map
CUDA_VISIBLE_DEVICES=4,5,6,7 torchpack dist-run -np 1 python tools/test.py configs/nuscenes/seg/fusion-bev256d2-rs50_depth_lss.yaml pretrained/bevfusion-seg.pth --eval map
