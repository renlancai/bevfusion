import mmcv
import torch


# def evaluate_map(self, results):
#     thresholds = torch.tensor([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])

#     num_classes = len(self.map_classes)
#     num_thresholds = len(thresholds)

#     tp = torch.zeros(num_classes, num_thresholds)
#     fp = torch.zeros(num_classes, num_thresholds)
#     fn = torch.zeros(num_classes, num_thresholds)

#     for result in results:
#         pred = result["masks_bev"]
#         label = result["gt_masks_bev"]

#         pred = pred.detach().reshape(num_classes, -1)
#         label = label.detach().bool().reshape(num_classes, -1)

#         pred = pred[:, :, None] >= thresholds
#         label = label[:, :, None]

#         tp += (pred & label).sum(dim=1)
#         fp += (pred & ~label).sum(dim=1)
#         fn += (~pred & label).sum(dim=1)

#     ious = tp / (tp + fp + fn + 1e-7)

#     metrics = {}
#     for index, name in enumerate(self.map_classes):
#         metrics[f"map/{name}/iou@max"] = ious[index].max().item()
#         for threshold, iou in zip(thresholds, ious[index]):
#             metrics[f"map/{name}/iou@{threshold.item():.2f}"] = iou.item()
#     metrics["map/mean/iou@max"] = ious.max(dim=1).values.mean().item()
#     return metrics

import numpy as np
import cv2
from sklearn.cluster import DBSCAN

def bev_seg_to_instances(bev_seg):
    """
    将BEV分割结果转换为实例级对象
    输入：
        bev_seg: [H, W] int型分割矩阵
        class_config: 类别配置字典，示例：
            {
                'lane': 1,    # 车道线类别ID
                'curb': 2,    # 路沿类别ID
                'min_area': {
                    'lane': 50,
                    'curb': 100
                }
            }
    输出：
        instances: 包含各实例信息的字典列表
    """
    instances = []
    
    class_config = {}
    class_config['divider'] = 1
    class_config['ped_crossing'] = 2
    class_config['boundary'] = 3
    class_config['min_area'] = {}
    class_config['min_area']['divider'] = 50
    class_config['min_area']['ped_crossing'] = 50
    class_config['min_area']['boundary'] = 50
    
    # map_classes = ["divider", "ped_crossing", "boundary"] # from Mask2Map
    # map_classes: # from bevfusion
    #     - drivable_area
    #     - ped_crossing
    #     - walkway
    #     - stop_line
    #     - carpark_area
    #     - divider
    
    for cls_name in ['divider', 'ped_crossing']:
        cls_id = class_config[cls_name]
        mask = (bev_seg == cls_id).astype(np.uint8)
        
        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        
        for label in range(1, num_labels):
            # 提取当前实例的像素坐标
            y, x = np.where(labels == label)
            
            # 计算几何特征
            area = stats[label, cv2.CC_STAT_AREA]
            width = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]
            
            # 面积过滤
            if area < class_config['min_area'][cls_name]:
                continue
                
            # 提取实例轮廓
            instance_mask = (labels == label).astype(np.uint8)
            contours, _ = cv2.findContours(instance_mask, 
                                         cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
            
            # 曲线拟合（适用于车道线）
            if cls_name == 'lane':
                poly = fit_lane_curve(x, y)  # 自定义曲线拟合函数
            else:
                poly = None
                
            instances.append({
                'class': cls_name,
                'points': np.column_stack((x, y)),
                'contour': contours.squeeze(),
                'centroid': centroids[label],
                'poly': poly,
                'geometry': {
                    'area': area,
                    'width': width,
                    'height': height
                }
            })
    
    return instances

def fit_lane_curve(x, y, order=3):
    """
    使用RANSAC拟合多项式曲线
    输入：
        x, y: 像素坐标点
        order: 多项式阶数
    输出：
        拟合后的多项式系数
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import RANSACRegressor
    
    # 转换坐标系（y为BEV空间前进方向）
    X = y.reshape(-1, 1)
    Y = x
    
    # 创建多项式回归模型
    model = make_pipeline(
        PolynomialFeatures(order),
        RANSACRegressor(min_samples=0.5)
    )
    
    model.fit(X, Y)
    return model.named_steps['ransacregressor'].estimator_.coef_

def single_gpu_test_val(model, data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        
        # for every result, we postprocess the seg mask to get vectorized instances
        pred = result["masks_bev"]
        label = result["gt_masks_bev"]

        
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results




