from mmdet.apis import inference_detector, init_detector
from pycocotools.coco import COCO
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import csv
import os

def compute_iou(box1, box2):
    """Calcula IoU entre duas caixas"""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = max(0, x2 - x1) * max(0, y2 - y1)
    box2_area = max(0, x2g - x1g) * max(0, y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

base_dir = "/mnt/hd_pesquisa/pesquisa/filipe/crcn_mm3x/"

# --- SETUP ---
#retinanet
#config_file = '/mnt/hd_pesquisa/pesquisa/camile/crcn_mm3x/configs/my_config/sota/micronucleo_reninanet_r50_fpn_1x_voc0712_20c_12x1ep.py'
config_file = os.path.join(base_dir,'configs/my_config/sota/micronucleo_reninanet_r50.py')
checkpoint_file = os.path.join(base_dir,'work_dirs/retinanet_baseline_lr001/epoch_200.pth')
output_dir = os.path.join(base_dir,'work_dirs/retinanet_baseline_lr001/')
#fcos
# config_file = '/mnt/hd_pesquisa/pesquisa/filipe/crcn_mm3x/configs/my_config/sota/micronucleo_fcos.py'
# checkpoint_file = '/mnt/hd_pesquisa/pesquisa/filipe/crcn_mm3x/work_dirs/fcos_baseline_1080_200ep/epoch_200.pth'



#config_file = '/mnt/hd_pesquisa/pesquisa/camile/crcn_mm3x/configs/my_config/sota/micronucleo_reninanet_r50_fpn_1x_voc0712_20c_12x1ep.py'

#checkpoint_file = '/mnt/hd_pesquisa/pesquisa/camile/crcn_mm3x/work_dirs/faster_baseline/retinaNet_crop_epochs_200/epoch_200.pth'

ann_file = '/mnt/hd_pesquisa/pesquisa/datasets/micronucleo_kaggle/annotations/test.json'
img_prefix = '/mnt/hd_pesquisa/pesquisa/datasets/micronucleo_kaggle/images/test/'

model = init_detector(config_file, checkpoint_file, device='cuda:0')
coco = COCO(ann_file)
cat_id_to_name = {k: v['name'] for k, v in coco.cats.items()}
class_names = [cat_id_to_name[i] for i in sorted(cat_id_to_name)]

y_true_all = []
y_pred_all = []




output_csv = os.path.join(output_dir,'predictions.csv')
print(output_csv)
#ok

csv_file = open(output_csv, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['image_name', 'class_id', 'score', 'x1', 'y1', 'x2', 'y2'])  # cabeçalho

# Iterar sobre as imagens
for img_id in coco.imgs:
    img_info = coco.loadImgs(img_id)[0]
    file_path = img_prefix + img_info['file_name']
    gt_ann_ids = coco.getAnnIds(imgIds=img_id)
    gt_anns = coco.loadAnns(gt_ann_ids)

    gt_boxes = [ann['bbox'] for ann in gt_anns]
    gt_labels = [ann['category_id'] for ann in gt_anns]

    result = inference_detector(model, file_path)

    det_boxes = result.pred_instances.bboxes
    det_labels = result.pred_instances.labels
    det_scores = result.pred_instances.scores

    pred_list = []
    for class_id, bbox, score in zip(det_labels, det_boxes, det_scores):
        if score < 0.3:  # filtro de confiança
            continue
        x1, y1, x2, y2 = bbox.cpu().numpy()
        pred_list.append(((x1, y1, x2, y2), class_id.item(), score.item()))

        # --- Salva a predição no CSV ---
        csv_writer.writerow([
            img_info['file_name'],
            class_id.item(),
            score.item(),
            x1, y1, x2, y2
        ])

    matched = set()

    for gt_box, gt_class in zip(gt_boxes, gt_labels):
        x, y, w, h = gt_box
        gt_coords = [x, y, x + w, y + h]
        best_iou = 0
        best_pred = None

        for i, (pred_box, pred_class, score) in enumerate(pred_list):
            iou = compute_iou(pred_box, gt_coords)
            if iou >= 0.5 and iou > best_iou:
                best_iou = iou
                best_pred = (pred_class, i)
        
        if best_pred:
            pred_class, pred_idx = best_pred
            matched.add(pred_idx)
            y_true_all.append(gt_class)
            y_pred_all.append(pred_class)
        else:
            # Falso negativo
            y_true_all.append(gt_class)
            y_pred_all.append(-1)

    # Falsos positivos restantes
    for i, (pred_box, pred_class, _) in enumerate(pred_list):
        if i not in matched:
            y_true_all.append(-1)
            y_pred_all.append(pred_class)

# Filtrar para apenas as classes válidas nas métricas
valid_labels = sorted(cat_id_to_name.keys())
y_true_filtered = []
y_pred_filtered = []

for yt, yp in zip(y_true_all, y_pred_all):
    if yt in valid_labels or yp in valid_labels:
        y_true_filtered.append(yt)
        y_pred_filtered.append(yp)

# Fechar CSV
csv_file.close()
print(f"Predições salvas em: {output_csv}")

# Calcular métricas
precision, recall, fscore, support = precision_recall_fscore_support(
    y_true_filtered, y_pred_filtered, labels=valid_labels, zero_division=0
)

# Mostrar resultados
for i, class_name in enumerate(class_names):
    print(f"Classe: {class_name}")
    print(f"  Precisão: {precision[i]:.3f}")
    print(f"  Recall: {recall[i]:.3f}")
    print(f"  F1-score: {fscore[i]:.3f}")
    print(f"  Support: {support[i]}")

