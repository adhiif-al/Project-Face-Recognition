import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import cv2
import threading
from backend import FaceRecognitionBackend
from visualisasi import VisualisasiWindow

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class WebcamWindow(ctk.CTkToplevel):
    """Live webcam preview window with a Capture button."""

    def __init__(self, master, on_capture_callback):
        super().__init__(master)
        self.title("Webcam – Capture Face")
        self.geometry("560x480")
        self.resizable(False, False)
        self.configure(fg_color="#09090b")
        self.on_capture_callback = on_capture_callback

        self._cap = None
        self._running = False
        self._last_frame = None

        # --- UI ---
        self.lbl_feed = ctk.CTkLabel(self, text="", fg_color="#111114")
        self.lbl_feed.pack(fill="both", expand=True, padx=15, pady=(15, 8))

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(pady=(0, 12))

        self.btn_capture = ctk.CTkButton(
            btn_row, text="📸  Capture & Recognize",
            fg_color="#3a86ff", hover_color="#0066cc",
            font=("Arial", 12, "bold"), height=34, width=200,
            command=self._capture
        )
        self.btn_capture.pack(side="left", padx=8)

        btn_cancel = ctk.CTkButton(
            btn_row, text="Cancel",
            fg_color="#2c2c2e", hover_color="#3a3a3c",
            font=("Arial", 12), height=34, width=100,
            command=self._close
        )
        btn_cancel.pack(side="left", padx=8)

        self.lbl_status = ctk.CTkLabel(self, text="Opening camera…",
                                       font=("Arial", 10), text_color="#636366")
        self.lbl_status.pack(pady=(0, 6))

        self.protocol("WM_DELETE_WINDOW", self._close)
        self._start_camera()

    # ------------------------------------------------------------------ camera

    def _start_camera(self):
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            self.lbl_status.configure(text="❌  Tidak dapat membuka kamera.", text_color="#ff453a")
            return
        self._running = True
        self.lbl_status.configure(text="Kamera aktif – arahkan wajah ke kamera.", text_color="#30d158")
        threading.Thread(target=self._feed_loop, daemon=True).start()

    def _feed_loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                break
            self._last_frame = frame
            # Convert BGR → RGB → PIL → CTkImage
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            ctk_img = ctk.CTkImage(pil_img, size=(520, 390))
            # Schedule UI update on main thread
            self.after(0, self._update_feed, ctk_img)
            # ~30 fps
            import time; time.sleep(0.033)

    def _update_feed(self, ctk_img):
        try:
            self.lbl_feed.configure(image=ctk_img)
        except Exception:
            pass

    # ------------------------------------------------------------------ actions

    def _capture(self):
        if self._last_frame is None:
            messagebox.showwarning("Webcam", "Belum ada frame yang diterima.", parent=self)
            return
        frame = self._last_frame.copy()
        self._close()
        self.on_capture_callback(frame)

    def _close(self):
        self._running = False
        if self._cap:
            self._cap.release()
        self.destroy()


# ============================================================== Main App ======

class FaceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Face Recognition System")
        self.geometry("1200x700")
        self.resizable(False, False)

        self.backend = FaceRecognitionBackend()
        self.test_img_path = None
        self._webcam_frame = None   # holds captured numpy frame when using webcam

        self.setup_layout()

    # ------------------------------------------------------------------ layout

    def setup_layout(self):
        # ===== SIDEBAR =====
        self.sidebar = ctk.CTkFrame(self, width=290, corner_radius=0, fg_color="#18181c")
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        lbl_title = ctk.CTkLabel(self.sidebar, text="Konfigurasi & Input Data",
                                  font=("Arial", 15, "bold"), text_color="#f5f5f7")
        lbl_title.pack(pady=(25, 15), padx=20, anchor="w")

        # --- 1. Dataset Input ---
        lbl_step1 = ctk.CTkLabel(self.sidebar, text="1. Dataset Input",
                                  font=("Arial", 12, "bold"), text_color="#f5f5f7")
        lbl_step1.pack(pady=(5, 2), padx=20, anchor="w")

        lbl_path_desc = ctk.CTkLabel(self.sidebar, text="Select folder containing dataset images:",
                                      font=("Arial", 10), text_color="#8e8e93")
        lbl_path_desc.pack(padx=20, anchor="w")

        # Choose Folder row (matches screenshot style)
        folder_row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        folder_row.pack(padx=20, pady=(6, 4), fill="x")

        self.btn_choose_folder = ctk.CTkButton(
            folder_row, text="Choose Folder", width=110, height=28,
            fg_color="#2c2c2e", hover_color="#3a3a3c",
            text_color="#f5f5f7", font=("Arial", 11),
            command=self.browse_dataset
        )
        self.btn_choose_folder.pack(side="left")

        self.lbl_folder_name = ctk.CTkLabel(
            folder_row, text="No Folder Chosen",
            font=("Arial", 10), text_color="#636366", anchor="w"
        )
        self.lbl_folder_name.pack(side="left", padx=8, fill="x", expand=True)

        # Dataset status indicator
        self.lbl_dataset_status = ctk.CTkLabel(
            self.sidebar, text="", font=("Arial", 9), text_color="#30d158"
        )
        self.lbl_dataset_status.pack(padx=20, anchor="w")

        # --- 2. Test Image ---
        lbl_step2 = ctk.CTkLabel(self.sidebar, text="2. Test Image",
                                  font=("Arial", 12, "bold"), text_color="#f5f5f7")
        lbl_step2.pack(pady=(12, 2), padx=20, anchor="w")

        lbl_test_desc = ctk.CTkLabel(self.sidebar, text="Choose test image for identification:",
                                      font=("Arial", 10), text_color="#8e8e93")
        lbl_test_desc.pack(padx=20, anchor="w")

        # Drop / Browse box
        self.drop_box = ctk.CTkFrame(self.sidebar, width=250, height=100,
                                      fg_color="#111114", border_width=1, border_color="#2c2c2e")
        self.drop_box.pack(pady=(4, 5), padx=20)
        self.drop_box.pack_propagate(False)

        lbl_drag = ctk.CTkLabel(self.drop_box,
                                 text="Drag and drop file here\nLimit 200MB per file • JPG, JPEG, PNG",
                                 font=("Arial", 9), text_color="#636366")
        lbl_drag.pack(pady=(12, 8))

        btn_browse = ctk.CTkButton(self.drop_box, text="Browse files", width=90, height=23,
                                   fg_color="#2c2c2e", hover_color="#3a3a3c",
                                   text_color="#f5f5f7", font=("Arial", 11),
                                   command=self.browse_file)
        btn_browse.pack()

        # Uploaded file label row
        self.file_container = ctk.CTkFrame(self.sidebar, fg_color="transparent", height=28)
        self.file_container.pack(padx=20, fill="x", pady=2)

        self.lbl_file_name = ctk.CTkLabel(self.file_container, text="",
                                           font=("Arial", 10), text_color="#409eff", anchor="w")
        self.lbl_file_name.pack(side="left", fill="x", expand=True)

        self.btn_clear_file = ctk.CTkButton(
            self.file_container, text="✕", width=15, height=15,
            fg_color="transparent", hover_color="#2c2c2e",
            text_color="#636366", font=("Arial", 10),
            command=self.clear_uploaded_file
        )

        # --- Threshold slider ---
        lbl_thresh_title = ctk.CTkLabel(self.sidebar,
                                         text="Ambang Batas Threshold (Euclidean Distance)",
                                         font=("Arial", 11), text_color="#e5e5ea")
        lbl_thresh_title.pack(pady=(12, 0), padx=20, anchor="w")

        self.lbl_thresh_val = ctk.CTkLabel(self.sidebar, text="5000",
                                            font=("Arial", 11, "bold"), text_color="#ff453a")
        self.lbl_thresh_val.pack(padx=20, anchor="center")

        self.slider_thresh = ctk.CTkSlider(self.sidebar, from_=1000, to=10000,
                                            number_of_steps=180, width=250, height=12)
        self.slider_thresh.configure(button_color="#ff453a", progress_color="#ff453a",
                                      button_hover_color="#ff6961", fg_color="#2c2c2e")
        self.slider_thresh.set(5000)
        self.slider_thresh.configure(command=self.on_slider_move)
        self.slider_thresh.pack(padx=20, pady=(2, 2))

        self.range_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent", height=12)
        self.range_frame.pack(padx=20, fill="x")
        ctk.CTkLabel(self.range_frame, text="1000", font=("Arial", 9), text_color="#636366").pack(side="left")
        ctk.CTkLabel(self.range_frame, text="10000", font=("Arial", 9), text_color="#636366").pack(side="right")

        # Webcam checkbox
        self.check_webcam = ctk.CTkCheckBox(self.sidebar, text="Use webcam",
                                             font=("Arial", 11), text_color="#e5e5ea",
                                             checkbox_width=16, checkbox_height=16,
                                             command=self.on_webcam_toggle)
        self.check_webcam.pack(pady=(15, 15), padx=20, anchor="w")

        # --- 3. Identification Process ---
        lbl_step3 = ctk.CTkLabel(self.sidebar, text="3. Identification Process",
                                  font=("Arial", 12, "bold"), text_color="#f5f5f7")
        lbl_step3.pack(pady=(5, 5), padx=20, anchor="w")

        self.btn_start = ctk.CTkButton(
            self.sidebar, text="Start Face Recognition",
            fg_color="#3a86ff", hover_color="#0066cc",
            text_color="white", font=("Arial", 12, "bold"),
            height=33, command=self.run_recognition
        )
        self.btn_start.pack(pady=5, padx=20, fill="x")

        self.btn_visualisasi = ctk.CTkButton(
            self.sidebar, text="📊 Lihat Visualisasi",
            fg_color="#1c1c1e", hover_color="#2c2c2e",
            text_color="#38bdf8", font=("Arial", 11),
            height=20,
            command=self.open_visualisasi
        )
        self.btn_visualisasi.pack(pady=(4, 5), padx=20, fill="x")

        # ===== WORKSPACE (right) =====
        self.workspace = ctk.CTkFrame(self, fg_color="#09090b", corner_radius=0)
        self.workspace.pack(side="right", fill="both", expand=True)

        self.panel_container = ctk.CTkFrame(self.workspace, fg_color="transparent")
        self.panel_container.pack(pady=(30, 15), padx=30, fill="both", expand=True)

        # Left panel – test face
        self.frame_test = ctk.CTkFrame(self.panel_container, fg_color="#121214",
                                        border_width=1, border_color="#1c1c1e")
        self.frame_test.pack(side="left", fill="both", expand=True, padx=10)
        self.lbl_img_test = ctk.CTkLabel(self.frame_test, text="")
        self.lbl_img_test.pack(fill="both", expand=True, pady=(10, 25))
        ctk.CTkLabel(self.frame_test, text="Your Test Face",
                     font=("Arial", 10), text_color="#8e8e93").pack(side="bottom", pady=8)

        # Right panel – result
        self.frame_result = ctk.CTkFrame(self.panel_container, fg_color="#121214",
                                          border_width=1, border_color="#1c1c1e")
        self.frame_result.pack(side="right", fill="both", expand=True, padx=10)
        self.lbl_img_result = ctk.CTkLabel(self.frame_result, text="")
        self.lbl_img_result.pack(fill="both", expand=True, pady=(10, 25))
        self.lbl_tag_result = ctk.CTkLabel(self.frame_result, text="Identified Face",
                                            font=("Arial", 10), text_color="#8e8e93")
        self.lbl_tag_result.pack(side="bottom", pady=8)

        # Status bar
        self.status_bar = ctk.CTkFrame(self.workspace, height=38, fg_color="#111827",
                                        corner_radius=6, border_width=1, border_color="#1f2937")
        self.status_bar.pack(fill="x", padx=40, pady=(0, 15))
        self.status_bar.pack_propagate(False)

        self.lbl_metrics = ctk.CTkLabel(
            self.status_bar,
            text="Accuracy Level: -- %  |  Computation Time: -- second",
            font=("Arial", 11, "bold"), text_color="#38bdf8"
        )
        self.lbl_metrics.pack(side="left", padx=20, fill="y")

        ctk.CTkLabel(self.workspace, text="Face Recognition System © 2025.",
                     font=("Arial", 9), text_color="#48484a").pack(side="bottom", pady=8)

    # ------------------------------------------------------------------ callbacks

    def on_slider_move(self, value):
        self.lbl_thresh_val.configure(text=str(int(value)))

    def on_webcam_toggle(self):
        """When webcam is checked, clear any loaded file (mutually exclusive input)."""
        if self.check_webcam.get():
            self.clear_uploaded_file()

    # ------------------------------------------------------------------ dataset

    def browse_dataset(self):
        folder = filedialog.askdirectory(title="Select Dataset Folder")
        if not folder:
            return
        self.backend.dataset_path = folder
        # Show truncated folder name
        display = os.path.basename(folder) or folder
        if len(display) > 22:
            display = "…" + display[-20:]
        self.lbl_folder_name.configure(text=display, text_color="#f5f5f7")
        self.lbl_dataset_status.configure(text="Training…", text_color="#ffd60a")
        self.update_idletasks()

        success = self.backend.train()
        if success:
            count = len(self.backend.image_names)
            self.lbl_dataset_status.configure(
                text=f"✓ {count} image(s) loaded", text_color="#30d158"
            )
        else:
            self.lbl_dataset_status.configure(
                text="❌ No valid images found", text_color="#ff453a"
            )

    # ------------------------------------------------------------------ test image

    def browse_file(self):
        file = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file:
            return
        # Uncheck webcam if a file is chosen
        self.check_webcam.deselect()
        self._webcam_frame = None
        self.test_img_path = file

        self.lbl_file_name.configure(text=f"📄 {os.path.basename(file)}")
        self.btn_clear_file.pack(side="right", padx=5)

        img = Image.open(file)
        img_ctk = ctk.CTkImage(img, size=(340, 340))
        self.lbl_img_test.configure(image=img_ctk, text="")

    def clear_uploaded_file(self):
        self.test_img_path = None
        self._webcam_frame = None
        self.lbl_file_name.configure(text="")
        self.btn_clear_file.pack_forget()
        self.lbl_img_test.configure(image=None, text="")
        self.lbl_img_result.configure(image=None, text="")
        self.lbl_tag_result.configure(text="Identified Face")

    # ------------------------------------------------------------------ webcam

    def open_webcam_window(self):
        win = WebcamWindow(self, on_capture_callback=self._on_webcam_capture)
        win.grab_set()

    def _on_webcam_capture(self, frame_bgr):
        """Called by WebcamWindow after the user presses Capture."""
        self._webcam_frame = frame_bgr
        self.test_img_path = None  # not a file

        # Show captured frame in the left panel
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        img_ctk = ctk.CTkImage(pil_img, size=(340, 340))
        self.lbl_img_test.configure(image=img_ctk, text="")
        self.lbl_file_name.configure(text="📷 Webcam capture")

        # Auto-run recognition after capture
        self._do_recognition()

    # ------------------------------------------------------------------ recognition

    def run_recognition(self):
        # Validate dataset
        if not self.backend.dataset_path or self.backend.eigenfaces is None:
            messagebox.showerror("Error", "Tentukan & muat folder dataset terlebih dahulu!")
            return

        if self.check_webcam.get():
            # Open webcam window; recognition runs after capture
            self.open_webcam_window()
        else:
            if not self.test_img_path:
                messagebox.showerror("Error", "Pilih foto wajah uji terlebih dahulu!")
                return
            self._do_recognition()
    
    def open_visualisasi(self):
        if self.backend.eigenfaces is None:
            messagebox.showwarning("Visualisasi", "Muat dataset terlebih dahulu!")
            return
        win = VisualisasiWindow(self, self.backend)
        win.grab_set()

    def _do_recognition(self):
        threshold = int(self.slider_thresh.get())

        if self._webcam_frame is not None:
            name, dist, acc, comp_time = self.backend.recognize_from_frame(
                self._webcam_frame, threshold=threshold
            )
        else:
            name, dist, acc, comp_time = self.backend.recognize(
                self.test_img_path, threshold=threshold
            )

        if name is None:
            messagebox.showwarning("Warning", "Gagal memproses perhitungan pengenalan wajah.")
            return

        self.lbl_metrics.configure(
            text=f"Accuracy Level: {acc:.2f}%  |  Computation Time: {comp_time:.4f} second"
        )

        if name == "Unknown":
            self.lbl_img_result.configure(image=None,
                                           text="⚠️ Wajah Tidak Dikenal\n(Melebihi Threshold)")
            self.lbl_tag_result.configure(text="Identified Face: Unknown")
        else:
            # Cari gambar referensi di subfolder yang namanya == name
            matched_path = None
            for root, dirs, files in os.walk(self.backend.dataset_path):
                if os.path.basename(root) == name:
                    for f in files:
                        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                            matched_path = os.path.join(root, f)
                            break
                if matched_path:
                    break  # ← break di sini, LALU display di luar loop

            # Tampilkan gambar — di LUAR for loop
            if matched_path:
                img_match = Image.open(matched_path)
                self._result_img = ctk.CTkImage(img_match, size=(340, 340))  # simpan di self!
                self.lbl_img_result.configure(image=self._result_img, text="")
                self.lbl_tag_result.configure(text=f"Identified Face: {name}")
            else:
                self.lbl_img_result.configure(image=None, text=f"Matched: {name}")
                self.lbl_tag_result.configure(text=f"Identified Face: {name}")

if __name__ == "__main__":
    app = FaceApp()
    app.mainloop()
