import os
import pickle
import numpy as np
import face_recognition
import kagglehub
from sklearn import svm
from sklearn.model_selection import train_test_split

def train_model():
    # 1. Tải dataset từ Kaggle
    print("--- Đang tải dataset từ Kaggle ---")
    path = kagglehub.dataset_download("vishesh1412/celebrity-face-image-dataset")
    
    # Đường dẫn thư mục chứa các folder nghệ sĩ
    dataset_dir = os.path.join(path, "Celebrity Faces Dataset")
    
    X = []
    y = []

    print("--- Đang trích xuất đặc trưng khuôn mặt (Feature Extraction) ---")
    for person_name in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_path):
            continue
            
        print(f"Đang xử lý: {person_name}")
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            try:
                image = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(image)
                if len(encodings) == 1:
                    X.append(encodings[0])
                    y.append(person_name)
            except:
                continue

    # 2. Huấn luyện mô hình
    print(f"--- Đang huấn luyện với {len(X)} mẫu dữ liệu ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = svm.SVC(gamma='scale', probability=True)
    clf.fit(X_train, y_train)
    
    # 3. Lưu mô hình
    os.makedirs("models", exist_ok=True)
    with open("models/face_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    print("--- Đã lưu mô hình vào models/face_classifier.pkl ---")

if __name__ == "__main__":
    train_model()