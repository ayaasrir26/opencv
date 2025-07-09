import cv2
import numpy as np

# ✅ تحميل كلاسيفاير ديال الوجه
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ✅ تحميل الفلاتر وموضعهم المناسب
filters = {
    '1': {
        'img': cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED),
        'y_offset_ratio': 0.32 
    },
    '2': {
        'img': cv2.imread("crown.png", cv2.IMREAD_UNCHANGED),
        'y_offset_ratio': -0.3  
    },

    '3': {
        'img': cv2.imread("glasses2.png", cv2.IMREAD_UNCHANGED),
        'y_offset_ratio': 0.32
    },
    '4': {
        'img': cv2.imread("pngwing.png", cv2.IMREAD_UNCHANGED),
        'y_offset_ratio': 0.50 
    },
    '5': {
        'img': cv2.imread("soleil.png", cv2.IMREAD_UNCHANGED),
        'y_offset_ratio': 0.50  
    }
}

# ✅ الفلتر الحالي
current_filter_key = '1'

# ✅ فتح الكاميرا
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ⬛ تحويل إلى رمادي
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 👤 كشف الوجه
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # 👓 جلب الفلتر الحالي
    filter_data = filters.get(current_filter_key, None)

    if filter_data is None or filter_data['img'] is None:
        continue

    filter_img = filter_data['img']
    y_offset_ratio = filter_data['y_offset_ratio']

    for (x, y, w, h) in faces:
        # 🖼️ تغيير حجم الفلتر
        filter_resized = cv2.resize(filter_img, (w, int(h / 3)))

        # 🧠 تحديد الموضع
        y1 = y + int(h * y_offset_ratio)
        y2 = y1 + filter_resized.shape[0]
        x1 = x
        x2 = x1 + filter_resized.shape[1]

        # ✅ تفادي الخروج عن الصورة
        if y1 < 0 or y2 > frame.shape[0] or x1 < 0 or x2 > frame.shape[1]:
            continue

        # 🔍 تقسيم الصورة لقنوات
        b, g, r, a = cv2.split(filter_resized)
        mask = cv2.merge((a, a, a))
        filter_rgb = cv2.merge((b, g, r))

        # 🧩 دمج الفلتر على الوجه
        roi = frame[y1:y2, x1:x2]
        roi_bg = cv2.bitwise_and(roi, 255 - mask)
        filter_fg = cv2.bitwise_and(filter_rgb, mask)
        dst = cv2.add(roi_bg, filter_fg)
        frame[y1:y2, x1:x2] = dst

    # ✅ كتابة التعليمات
    cv2.putText(frame, "Press 1-6 to change filter | Press 'q' to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # ✅ عرض الصورة
    cv2.imshow('Snapchat Filters - OpenCV', frame)

    # 🎮 التحكم بالأزرار
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif chr(key) in filters:
        current_filter_key = chr(key)

# ❌ إغلاق الكاميرا والنوافذ
cap.release()
cv2.destroyAllWindows() 