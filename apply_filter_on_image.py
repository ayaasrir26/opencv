import cv2
import numpy as np

#  تحميل كلاسيفاير الوجه
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# تحميل الصورة
img = cv2.imread("photo.jpg")
print("img =", img)

img_original = img.copy()

#  تحويل لـ Grayscale للكشف
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#  كشف الوجوه
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

#  تحميل الفلتر (مثلاً نظارات)
filter_img = cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED)
y_offset_ratio = 0.35  # نظارات = فوق العينين

#  لكل وجه فالصورة، نركب الفلتر
for (x, y, w, h) in faces:
    filter_resized = cv2.resize(filter_img, (w, int(h / 3)))

    y1 = y + int(h * y_offset_ratio)
    y2 = y1 + filter_resized.shape[0]
    x1 = x
    x2 = x1 + filter_resized.shape[1]

    if y1 < 0 or y2 > img.shape[0] or x1 < 0 or x2 > img.shape[1]:
        continue

    b, g, r, a = cv2.split(filter_resized)
    mask = cv2.merge((a, a, a))
    filter_rgb = cv2.merge((b, g, r))

    roi = img[y1:y2, x1:x2]
    roi_bg = cv2.bitwise_and(roi, 255 - mask)
    filter_fg = cv2.bitwise_and(filter_rgb, mask)
    dst = cv2.add(roi_bg, filter_fg)

    img[y1:y2, x1:x2] = dst

#  عرض النتيجة
cv2.imshow("Original Image", img_original)
cv2.imshow("Image with Filter", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
