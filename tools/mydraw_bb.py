import os
import cv2
import pandas as pd

# Caminho para as imagens e pasta de saída
image_folder = "/Users/filipe/Downloads/micronucleo/images/test"
#output_folder = "./images"
output_folder = "/Users/filipe/Desktop/artigo_micronucleo/images"
os.makedirs(output_folder, exist_ok=True)

# Cores e fonte
bbox_color = (0, 255, 0)  # verde
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 3
font_thickness = 4

# Carrega o CSV
df = pd.read_csv(os.path.join("/Users/filipe/Desktop/artigo_micronucleo","test.csv"))
df_pred = pd.read_csv(os.path.join("/Users/filipe/Desktop/artigo_micronucleo","predictions.csv"))

# Agrupa por imagem
for filename, group in df.groupby("filename"):
    image_path = os.path.join(image_folder, filename)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Imagem não encontrada: {image_path}")
        continue

    for _, row in group.iterrows():
        xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        label = str(row['label'])
        # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # cv2.putText(image, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # Caixa do bounding box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), bbox_color, 12)

        # Texto e fundo do texto
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        text_x = xmin
        text_y = ymin - 5
        cv2.rectangle(image, (text_x, text_y - text_h - baseline), (text_x + text_w, text_y), bbox_color, -1)
        # cv2.rectangle(image, (text_x, text_y - text_h - baseline), (text_x + text_w, text_y), bbox_color, 4)
        cv2.putText(image, label, (text_x, text_y - 2), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, image)