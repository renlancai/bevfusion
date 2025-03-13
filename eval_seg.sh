#CUDA_VISIBLE_DEVICES=1,2 torchpack dist-run -np 2 python tools/test.py configs/nuscenes/seg/fusion-bev256d2-lss.yaml pretrained/bevfusion-seg.pth --eval map
CUDA_VISIBLE_DEVICES=0,1 torchpack dist-run -np 2 python tools/test.py configs/nuscenes/seg/fusion-bev256d2-lss.yaml pretrained/bevfusion-seg.pth --eval map
