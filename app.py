import streamlit as st
import face_recognition
import pickle
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="AI Face Recognition", layout="centered")

st.title("👤 Hệ thống Nhận diện Khuôn mặt Nghệ sĩ")
st.write("Upload một tấm ảnh để AI đoán xem đó là ai!")

# 1. Load mô hình đã train
@st.cache_resource
def load_model():
    try:
        with open("models/face_classifier.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

model = load_model()

if model is None:
    st.error("Chưa tìm thấy file mô hình! Hãy chạy 'python train.py' trước.")
else:
    # 2. Giao diện Upload
    uploaded_file = st.file_uploader("Chọn ảnh (jpg, png, jpeg)...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Chuyển file upload sang định dạng ảnh OpenCV
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Streamlit dùng RGB, nhưng OpenCV cần BGR để vẽ, sau đó hiện lại RGB
        display_img = image_np.copy()
        
        # 3. Xử lý nhận diện
        face_locations = face_recognition.face_locations(image_np)
        face_encodings = face_recognition.face_encodings(image_np, face_locations)

        if len(face_encodings) == 0:
            st.warning("Không tìm thấy khuôn mặt nào trong ảnh.")
        else:
            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                # Dự đoán
                name = model.predict([encoding])[0]
                probs = model.predict_proba([encoding])
                confidence = np.max(probs) * 100

                # Vẽ khung lên ảnh
                color = (0, 255, 0) if confidence > 50 else (255, 0, 0)
                cv2.rectangle(display_img, (left, top), (right, bottom), color, 4)
                cv2.putText(display_img, f"{name} ({confidence:.1f}%)", 
                            (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Hiển thị kết quả
            st.image(display_img, caption="Kết quả nhận diện", use_column_width=True)
            st.success(f"Tìm thấy {len(face_encodings)} khuôn mặt!")