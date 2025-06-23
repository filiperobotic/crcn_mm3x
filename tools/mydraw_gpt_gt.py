import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import box as shapely_box
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def compute_iou(boxA, boxB):
    bA = shapely_box(*boxA)
    bB = shapely_box(*boxB)
    inter = bA.intersection(bB).area
    union = bA.union(bB).area
    return inter / union if union > 0 else 0

# Classe para nome (substitua conforme seu mapeamento)
class_id_to_name = {
    0: "BN",
    1: "BNMN",
    2: "MN",
}

# Caminhos
img_dir = "/Users/filipe/Downloads/micronucleo/images/test"
#pred_csv = "/Users/filipe/Desktop/artigo_micronucleo/predictions_fcos.csv"
# pred_csv = "/Users/filipe/Desktop/artigo_micronucleo/predictions_retinanet.csv"
pred_csv = "/Users/filipe/Desktop/artigo_micronucleo/test.csv"
gt_csv = "/Users/filipe/Desktop/artigo_micronucleo/test.csv"
output_dir = "/Users/filipe/Desktop/artigo_micronucleo/imagens_com_caixas_gt"
output_confusion_dir = "/Users/filipe/Desktop/artigo_micronucleo/"
os.makedirs(output_dir, exist_ok=True)

# Leitura dos arquivos CSV
pred_df = pd.read_csv(pred_csv)
gt_df = pd.read_csv(gt_csv)

grouped_gt = gt_df.groupby("filename")
# grouped_pred = pred_df.groupby("image_name")
grouped_pred = pred_df.groupby("filename")
# image_names = set(gt_df["filename"]).union(pred_df["image_name"])
image_names = set(gt_df["filename"]).union(pred_df["filename"])

# Fonte
FONT_SIZE = 24
try:
    font = ImageFont.truetype("arial.ttf", FONT_SIZE)
except IOError:
    # font = ImageFont.load_default()
    font = ImageFont.truetype("/Library/Fonts/Arial.ttf", FONT_SIZE)

def draw_label(draw, box, text, color, font):
    text_size = draw.textbbox((0, 0), text, font=font)
    text_width = text_size[2] - text_size[0]
    text_height = text_size[3] - text_size[1]
    x1, y1 = int(box[0]), int(box[1]) - text_height - 4
    x2, y2 = x1 + text_width + 6, y1 + text_height + 4
    draw.rectangle([x1, y1, x2, y2], fill=color)
    draw.text((x1 + 3, y1 + 2), text, fill="white", font=font)

# Coleta para matriz de confusão
y_true_all = []
y_pred_all = []

for image_name in image_names:
    image_path = os.path.join(img_dir, image_name)
    if not os.path.exists(image_path):
        print(f"Imagem não encontrada: {image_name}")
        continue

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    gts = grouped_gt.get_group(image_name) if image_name in grouped_gt.groups else pd.DataFrame()
    preds = grouped_pred.get_group(image_name) if image_name in grouped_pred.groups else pd.DataFrame()
    gt_used = set()

    for _, pred in preds.iterrows():
        #pred_box = (pred['x1'], pred['y1'], pred['x2'], pred['y2'])
        pred_box = (pred['xmin'], pred['ymin'], pred['xmax'], pred['ymax'])
        #pred_class = pred['class_id']
        pred_class = pred['label']
        # score = pred['score']
        score = 1
        matched = False

        for gt_idx, gt in gts.iterrows():
            if gt_idx in gt_used:
                continue
            gt_box = (gt['xmin'], gt['ymin'], gt['xmax'], gt['ymax'])
            gt_class = gt['label']
            iou = compute_iou(pred_box, gt_box)
            if iou >= 0.5 and pred_class == gt_class:
                draw.rectangle(pred_box, outline="green", width=10)
                text = f"{class_id_to_name.get(pred_class, str(pred_class))} {score:.2f}"
                draw_label(draw, pred_box, text, "green", font)
                y_true_all.append(gt_class)
                y_pred_all.append(pred_class)
                gt_used.add(gt_idx)
                matched = True
                break

        if not matched:
            draw.rectangle(pred_box, outline="red", width=10)
            text = f"{class_id_to_name.get(pred_class, str(pred_class))} {score:.2f}"
            draw_label(draw, pred_box, text, "red", font)
            y_true_all.append(-1)
            y_pred_all.append(pred_class)

    for gt_idx, gt in gts.iterrows():
        if gt_idx not in gt_used:
            gt_box = (gt['xmin'], gt['ymin'], gt['xmax'], gt['ymax'])
            gt_class = gt['label']
            draw.rectangle(gt_box, outline="blue", width=10)
            text = f"{class_id_to_name.get(gt_class, str(gt_class))}"
            draw_label(draw, gt_box, text, "blue", font)
            y_true_all.append(gt_class)
            y_pred_all.append(-1)


    image.save(os.path.join(output_dir, image_name))

print(f"Imagens geradas com rótulos em: {output_dir}")

# Gerar matriz de confusão
labels = sorted(set(y_true_all + y_pred_all) - {-1})
cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[class_id_to_name[i] for i in labels])
disp.plot(xticks_rotation=45, cmap='Blues')
plt.title("Matriz de Confusão")
plt.tight_layout()

# Salvar imagem da matriz
confusion_path = os.path.join(output_confusion_dir, "matriz_confusao.png")
plt.savefig(confusion_path)
plt.close()
print(f"Matriz de confusão salva em: {confusion_path}")