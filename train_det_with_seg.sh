# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchpack dist-run -np 1 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
#     --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
#     --run-dir work_dir/


CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchpack dist-run -np 1 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml  \
    --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
    --load_from pretrained/lidar-only-det.pth \
    --run-dir work_swint_v0p075/

# 30 machine seg with seg
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchpack dist-run -np 7 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50/convfuser.yaml \
#     --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/resnet50-0676ba61.pth \
#     --run-dir work_dir30/ \
    # --resume_from work_dir30/temporal_epoch_6.pth \

# 30 machine seg 
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchpack dist-run -np 1 python tools/train.py configs/nuscenes/seg/fusion-bev256d2-rs50_depth_lss.yaml \
#     --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/resnet50-0676ba61.pth \
#     --run-dir work_dir30/ \
    # --load_from pretrained/lidar-only-det.pth \


# 47 machine
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchpack dist-run -np 1 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50/convfuser.yaml \
#     --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/resnet50-0676ba61.pth \
#     --run-dir work_dir47/
