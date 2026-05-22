import os
import cv2
import numpy as np
import time

class FaceRecognitionBackend:
    def __init__(self, dataset_path="", size=(100, 100)):
        self.dataset_path = dataset_path
        self.size = size
        self.image_names = []
        self.mean_face = None
        self.eigenfaces = None
        self.weights = None

    def train(self):
        if not self.dataset_path or not os.path.exists(self.dataset_path):
            return False
        
        images = []
        self.image_names = []
        
        try:

            for root, dirs, files in os.walk(self.dataset_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                        path = os.path.join(root, file)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, self.size)
                            images.append(img.flatten())
                            # Gunakan nama subfolder sebagai label identitas
                            folder_name = os.path.basename(root)
                            self.image_names.append(folder_name)
                        
            if len(images) == 0:
                return False
                
            faces = np.array(images)
            self.mean_face = np.mean(faces, axis=0)
            centered_faces = faces - self.mean_face
            
            # Singular Value Decomposition (SVD) untuk mencari Eigenfaces
            U, S, Vt = np.linalg.svd(centered_faces.T, full_matrices=False)
            self.eigenfaces = U
            self.weights = np.dot(self.eigenfaces.T, centered_faces.T).T
            return True
        except Exception:
            return False

    def _project_and_match(self, gray_flat, threshold):
        """Shared projection + matching logic."""
        centered_test = gray_flat - self.mean_face
        test_weight = np.dot(self.eigenfaces.T, centered_test)
        distances = np.linalg.norm(self.weights - test_weight, axis=1)
        best_match_idx = np.argmin(distances)
        min_distance = distances[best_match_idx]

        if min_distance > threshold:
            return "Unknown", min_distance, 0.0

        accuracy = (1 - (min_distance / threshold)) * 100
        accuracy = max(0.0, min(100.0, accuracy))
        return self.image_names[best_match_idx], min_distance, accuracy

    def recognize(self, test_img_path, threshold=5000):
        if self.eigenfaces is None or self.mean_face is None:
            return None, 0, 0, 0
            
        start_time = time.time()
        
        img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, 0, 0, 0
            
        img = cv2.resize(img, self.size).flatten()
        name, dist, acc = self._project_and_match(img, threshold)
        comp_time = time.time() - start_time
        return name, dist, acc, comp_time

    def recognize_from_frame(self, frame_bgr, threshold=5000):
        """Recognize directly from a BGR numpy array (e.g. from cv2.VideoCapture)."""
        if self.eigenfaces is None or self.mean_face is None:
            return None, 0, 0, 0

        start_time = time.time()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, self.size).flatten()
        name, dist, acc = self._project_and_match(gray, threshold)
        comp_time = time.time() - start_time
        return name, dist, acc, comp_time
