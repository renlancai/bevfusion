# torchpack dist-run -np 1 python tools/test.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml pretrained/bevfusion-det.pth --eval bbox
torchpack dist-run -np 8 python tools/test.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml work_dir/whole_det.pth --eval bbox

#CUDA_VISIBLE_DEVICES=4,5,6,7 torchpack dist-run -np 1 python tools/test.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50/convfuser.yaml pretrained/qat/bevfusion_ptq.pth --eval bbox