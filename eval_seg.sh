#CUDA_VISIBLE_DEVICES=1,2 torchpack dist-run -np 2 python tools/test.py configs/nuscenes/seg/fusion-bev256d2-lss.yaml pretrained/bevfusion-seg.pth --eval map
CUDA_VISIBLE_DEVICES=4,5,6,7 torchpack dist-run -np 4 python tools/test.py configs/nuscenes/seg/fusion-bev256d2-lss.yaml pretrained/bevfusion-seg.pth --eval map
