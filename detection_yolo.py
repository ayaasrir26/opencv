#vedio
from ultralytics import YOLO
import cv2

# Charger le modèle YOLOv8 pré-entraîné
model = YOLO('yolov8m.pt')  # modèle plus précis

# Ouvrir la webcam (0 pour la webcam principale)
cap = cv2.VideoCapture(0)

# Liste des objets d'intérêt personnalisés
target_objects = [
    'person', 'bottle', 'cup', 'wine glass', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'clock',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
    'backpack', 'handbag', 'suitcase', 'tie',
    'bicycle', 'motorcycle', 'car', 'bus', 'train', 'truck',
    'pencil', 'notebook', 'paper', 'eraser', 'marker', 'crayon',
    'paintbrush', 'ruler', 'glue stick', 'tape dispenser',
    'stapler', 'paper clip', 'binder', 'folder', 'envelope', 'post-it notes',
    'whiteboard', 'chalkboard', 'calculator', 'computer mouse',
    'monitor', 'printer', 'scanner', 'projector', 'speaker', 'headphones', 'webcam'
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Appliquer le modèle YOLO sur l'image
    results = model.predict(frame, conf=0.4)
    annotated_frame = results[0].plot()

    # Parcourir les objets détectés
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])

            if cls_name in target_objects:
                print(f"✅ Objet détecté : {cls_name} avec confiance {conf:.2f}")

    # Affichage de l'image annotée
    cv2.imshow("Détection avec YOLOv8", annotated_frame)

    # Appuyer sur 'q' pour quitter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libération des ressources
cap.release()
cv2.destroyAllWindows()



# ##image

# # from ultralytics import YOLO
# # import cv2

# # def detect_objects_in_image(image_path, output_path="output.jpg"): 
# #     model = YOLO('yolov8m.pt')  
    
# #     target_objects = [
# #         'person', 'bottle', 'cup', 'wine glass', 'fork', 'knife', 'spoon', 'bowl',
# #         'banana', 'apple', 'orange', 'broccoli', 'carrot',
# #         'hot dog', 'pizza', 'donut', 'cake',
# #         'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
# #         'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'clock',
# #         'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
# #         'book', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
# #         'backpack', 'handbag', 'suitcase', 'tie',
# #         'bicycle', 'motorcycle', 'car', 'bus', 'train', 'truck',
# #         'pencil', 'notebook', 'paper', 'eraser', 'marker', 'crayon',
# #         'paintbrush', 'ruler', 'glue stick', 'tape dispenser',
# #         'stapler', 'paper clip', 'binder', 'folder', 'envelope', 'post-it notes',
# #         'whiteboard', 'chalkboard', 'calculator', 'computer mouse',
# #         'monitor', 'printer', 'scanner', 'projector', 'speaker', 'headphones', 'webcam'
# #     ]
# #     try:
# #         img = cv2.imread(image_path)
# #         if img is None:
# #             raise FileNotFoundError
# #     except:
# #         print(f"Erreur: Impossible de charger l'image à partir de {image_path}")
# #         return

    
# #     results = model.predict(img, conf=0.4) 
# #     for r in results:
# #         for box in r.boxes:
# #             cls_id = int(box.cls[0])
# #             cls_name = model.names[cls_id]
# #             conf = float(box.conf[0])
# #             if cls_name in target_objects:
# #                 print(f"Objet détecté : {cls_name} (confiance: {conf:.2f})")

    
# #     annotated_img = results[0].plot()
# #     cv2.imwrite(output_path, annotated_img)
# #     print(f"Résultats sauvegardés dans {output_path}")
# #     cv2.imshow("Résultats de détection", annotated_img)
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()

# # detect_objects_in_image("image/bird[1].png", "resultat.jpg")


