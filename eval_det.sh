CUDA_VISIBLE_DEVICES=4,5,6,7 torchpack dist-run -np 4 python tools/test.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml pretrained/bevfusion-det.pth --eval bbox

