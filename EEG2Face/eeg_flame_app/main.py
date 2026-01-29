import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QComboBox,
                             QSlider, QGroupBox, QFileDialog, QProgressBar,
                             QSpinBox, QDoubleSpinBox, QTabWidget, QTextEdit,
                             QCheckBox, QSplitter, QFrame, QLineEdit,
                             QMessageBox, QStatusBar)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor
import pyvistaqt
import pyvista as pv
from pathlib import Path
import time

from eeg_sources import SimulatedEEGSource, create_eeg_source
from flame_model import EEGToFLAMEModel


class EEGCollectionThread(QThread):
    data_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, eeg_source, interval_ms=100):
        super().__init__()
        self.eeg_source = eeg_source
        self.interval_ms = interval_ms
        self.running = False
        
    def run(self):
        self.running = True
        try:
            self.eeg_source.start()
            while self.running:
                data = self.eeg_source.get_sample()
                self.data_ready.emit(data)
                time.sleep(self.interval_ms / 1000.0)
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            try:
                self.eeg_source.stop()
            except:
                pass
            
    def stop(self):
        self.running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG to FLAME 3D Face Generator")
        self.setGeometry(100, 100, 1400, 900)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EEGToFLAMEModel().to(self.device)
        self.model.eval()
        
        self.eeg_source = SimulatedEEGSource()
        self.eeg_thread = None
        self.eeg_buffer = []
        self.is_collecting = False
        
        self.current_vertices = None
        self.current_faces = None
        self.current_exp_params = None
        self.current_pose_params = None
        
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0
        
        self._init_ui()
        
    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        eeg_group = QGroupBox("EEG Collection")
        eeg_layout = QVBoxLayout(eeg_group)
        
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Source:"))
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Simulated", "LSL Stream", "BrainFlow", "File"])
        self.source_combo.currentTextChanged.connect(self._on_source_changed)
        source_layout.addWidget(self.source_combo)
        eeg_layout.addLayout(source_layout)
        
        self.file_widget = QWidget()
        file_layout = QHBoxLayout(self.file_widget)
        file_layout.setContentsMargins(0, 0, 0, 0)
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select EEG file...")
        file_layout.addWidget(self.file_path_edit)
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self._browse_file)
        file_layout.addWidget(self.browse_btn)
        self.file_widget.hide()
        eeg_layout.addWidget(self.file_widget)
        
        emotion_layout = QHBoxLayout()
        emotion_layout.addWidget(QLabel("Simulated Emotion:"))
        self.emotion_combo = QComboBox()
        self.emotion_combo.addItems(["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"])
        self.emotion_combo.currentTextChanged.connect(self._on_emotion_changed)
        emotion_layout.addWidget(self.emotion_combo)
        eeg_layout.addLayout(emotion_layout)
        
        intensity_layout = QHBoxLayout()
        intensity_layout.addWidget(QLabel("Intensity:"))
        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setRange(0, 100)
        self.intensity_slider.setValue(50)
        self.intensity_slider.valueChanged.connect(self._on_intensity_changed)
        intensity_layout.addWidget(self.intensity_slider)
        self.intensity_label = QLabel("50%")
        intensity_layout.addWidget(self.intensity_label)
        eeg_layout.addLayout(intensity_layout)
        
        update_layout = QHBoxLayout()
        update_layout.addWidget(QLabel("Update Rate (ms):"))
        self.update_rate_spin = QSpinBox()
        self.update_rate_spin.setRange(50, 1000)
        self.update_rate_spin.setValue(100)
        update_layout.addWidget(self.update_rate_spin)
        eeg_layout.addLayout(update_layout)
        
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Collection")
        self.start_btn.clicked.connect(self._toggle_collection)
        btn_layout.addWidget(self.start_btn)
        self.process_btn = QPushButton("Process Single")
        self.process_btn.clicked.connect(self._process_single)
        btn_layout.addWidget(self.process_btn)
        eeg_layout.addLayout(btn_layout)
        
        self.status_label = QLabel("Status: Idle")
        eeg_layout.addWidget(self.status_label)
        
        self.fps_label = QLabel("FPS: 0")
        eeg_layout.addWidget(self.fps_label)
        
        left_layout.addWidget(eeg_group)
        
        face_group = QGroupBox("Face Parameters")
        face_layout = QVBoxLayout(face_group)
        
        shape_layout = QHBoxLayout()
        shape_layout.addWidget(QLabel("Shape:"))
        self.shape_slider = QSlider(Qt.Horizontal)
        self.shape_slider.setRange(-100, 100)
        self.shape_slider.setValue(0)
        self.shape_slider.valueChanged.connect(self._on_params_changed)
        shape_layout.addWidget(self.shape_slider)
        self.shape_value_label = QLabel("0")
        shape_layout.addWidget(self.shape_value_label)
        face_layout.addLayout(shape_layout)
        
        exp_scale_layout = QHBoxLayout()
        exp_scale_layout.addWidget(QLabel("Expression Scale:"))
        self.exp_scale_slider = QSlider(Qt.Horizontal)
        self.exp_scale_slider.setRange(0, 200)
        self.exp_scale_slider.setValue(100)
        exp_scale_layout.addWidget(self.exp_scale_slider)
        self.exp_scale_label = QLabel("100%")
        self.exp_scale_slider.valueChanged.connect(lambda v: self.exp_scale_label.setText(f"{v}%"))
        exp_scale_layout.addWidget(self.exp_scale_label)
        face_layout.addLayout(exp_scale_layout)
        
        self.auto_update_check = QCheckBox("Auto-update visualization")
        self.auto_update_check.setChecked(True)
        face_layout.addWidget(self.auto_update_check)
        
        self.smooth_check = QCheckBox("Smooth transitions")
        self.smooth_check.setChecked(True)
        face_layout.addWidget(self.smooth_check)
        
        left_layout.addWidget(face_group)
        
        render_group = QGroupBox("Rendering")
        render_layout = QVBoxLayout(render_group)
        
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Face Color:"))
        self.color_combo = QComboBox()
        self.color_combo.addItems(["Peach", "Gray", "Blue", "Custom"])
        self.color_combo.currentTextChanged.connect(self._update_mesh)
        color_layout.addWidget(self.color_combo)
        render_layout.addLayout(color_layout)
        
        self.wireframe_check = QCheckBox("Show Wireframe")
        self.wireframe_check.stateChanged.connect(self._update_mesh)
        render_layout.addWidget(self.wireframe_check)
        
        self.smooth_shading_check = QCheckBox("Smooth Shading")
        self.smooth_shading_check.setChecked(True)
        self.smooth_shading_check.stateChanged.connect(self._update_mesh)
        render_layout.addWidget(self.smooth_shading_check)
        
        left_layout.addWidget(render_group)
        
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)
        
        self.export_obj_btn = QPushButton("Export as OBJ")
        self.export_obj_btn.clicked.connect(self._export_obj)
        export_layout.addWidget(self.export_obj_btn)
        
        self.export_ply_btn = QPushButton("Export as PLY")
        self.export_ply_btn.clicked.connect(self._export_ply)
        export_layout.addWidget(self.export_ply_btn)
        
        self.export_params_btn = QPushButton("Export Parameters")
        self.export_params_btn.clicked.connect(self._export_params)
        export_layout.addWidget(self.export_params_btn)
        
        self.save_model_btn = QPushButton("Save Model Weights")
        self.save_model_btn.clicked.connect(self._save_model)
        export_layout.addWidget(self.save_model_btn)
        
        self.load_model_btn = QPushButton("Load Model Weights")
        self.load_model_btn.clicked.connect(self._load_model)
        export_layout.addWidget(self.load_model_btn)
        
        left_layout.addWidget(export_group)
        
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        log_layout.addWidget(self.log_text)
        left_layout.addWidget(log_group)
        
        left_layout.addStretch()
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.plotter_widget = pyvistaqt.QtInteractor(right_panel)
        self.plotter_widget.set_background('white')
        right_layout.addWidget(self.plotter_widget)
        
        view_layout = QHBoxLayout()
        self.front_btn = QPushButton("Front")
        self.front_btn.clicked.connect(lambda: self._set_view('front'))
        view_layout.addWidget(self.front_btn)
        self.side_btn = QPushButton("Side")
        self.side_btn.clicked.connect(lambda: self._set_view('side'))
        view_layout.addWidget(self.side_btn)
        self.top_btn = QPushButton("Top")
        self.top_btn.clicked.connect(lambda: self._set_view('top'))
        view_layout.addWidget(self.top_btn)
        self.reset_btn = QPushButton("Reset View")
        self.reset_btn.clicked.connect(lambda: self._set_view('reset'))
        view_layout.addWidget(self.reset_btn)
        self.screenshot_btn = QPushButton("Screenshot")
        self.screenshot_btn.clicked.connect(self._take_screenshot)
        view_layout.addWidget(self.screenshot_btn)
        right_layout.addLayout(view_layout)
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 1050])
        
        main_layout.addWidget(splitter)
        
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
        self._init_mesh()
        self._log("Application initialized")
        self._log(f"Using device: {self.device}")
        
    def _on_source_changed(self, source):
        self.file_widget.setVisible(source == "File")
        
    def _browse_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select EEG File", "", 
            "EEG Files (*.npy *.npz *.csv *.pt);;All Files (*)"
        )
        if filename:
            self.file_path_edit.setText(filename)
        
    def _init_mesh(self):
        with torch.no_grad():
            dummy_eeg = torch.randn(1, 7, 30).to(self.device)
            vertices, faces, _, _ = self.model(dummy_eeg)
            self.current_vertices = vertices[0].cpu().numpy()
            self.current_faces = faces.cpu().numpy()
        self._update_mesh()
        
    def _update_mesh(self):
        if self.current_vertices is None:
            return
            
        self.plotter_widget.clear()
        
        n_faces = len(self.current_faces)
        faces_pv = np.hstack([np.full((n_faces, 1), 3), self.current_faces]).flatten()
        
        mesh = pv.PolyData(self.current_vertices, faces_pv)
        mesh.compute_normals(inplace=True)
        
        color_map = {
            "Peach": "peachpuff",
            "Gray": "lightgray", 
            "Blue": "lightblue",
            "Custom": "wheat"
        }
        color = color_map.get(self.color_combo.currentText(), "peachpuff")
        
        show_edges = self.wireframe_check.isChecked()
        smooth = self.smooth_shading_check.isChecked()
        
        self.plotter_widget.add_mesh(
            mesh, 
            color=color, 
            smooth_shading=smooth,
            show_edges=show_edges, 
            lighting=True, 
            specular=0.5,
            edge_color='gray' if show_edges else None
        )
        
        self.plotter_widget.add_light(pv.Light(position=(1, 1, 1), intensity=0.8))
        self.plotter_widget.add_light(pv.Light(position=(-1, 0, 1), intensity=0.3))
        
    def _set_view(self, view):
        if view == 'front':
            self.plotter_widget.view_yz()
        elif view == 'side':
            self.plotter_widget.view_xz()
        elif view == 'top':
            self.plotter_widget.view_xy()
        else:
            self.plotter_widget.reset_camera()
            
    def _toggle_collection(self):
        if not self.is_collecting:
            self._start_collection()
        else:
            self._stop_collection()
            
    def _start_collection(self):
        source_type = self.source_combo.currentText().lower().replace(" ", "_")
        
        try:
            if source_type == "simulated":
                self.eeg_source = SimulatedEEGSource()
            elif source_type == "lsl_stream":
                self.eeg_source = create_eeg_source('lsl')
            elif source_type == "brainflow":
                self.eeg_source = create_eeg_source('brainflow', board_name='synthetic')
            elif source_type == "file":
                filepath = self.file_path_edit.text()
                if not filepath:
                    QMessageBox.warning(self, "Error", "Please select an EEG file first")
                    return
                self.eeg_source = create_eeg_source('file', filepath=filepath)
            else:
                self.eeg_source = SimulatedEEGSource()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create EEG source: {str(e)}")
            return
            
        self.is_collecting = True
        self.start_btn.setText("Stop Collection")
        self.status_label.setText("Status: Collecting...")
        
        self.eeg_thread = EEGCollectionThread(self.eeg_source, self.update_rate_spin.value())
        self.eeg_thread.data_ready.connect(self._on_eeg_data)
        self.eeg_thread.error_occurred.connect(self._on_collection_error)
        self.eeg_thread.start()
        
        self._log(f"Started EEG collection from {source_type}")
        
    def _stop_collection(self):
        self.is_collecting = False
        self.start_btn.setText("Start Collection")
        self.status_label.setText("Status: Idle")
        
        if self.eeg_thread:
            self.eeg_thread.stop()
            self.eeg_thread.wait()
            self.eeg_thread = None
            
        self._log("Stopped EEG collection")
        
    def _on_collection_error(self, error_msg):
        self._stop_collection()
        QMessageBox.critical(self, "Collection Error", error_msg)
        self._log(f"Error: {error_msg}")
        
    def _on_eeg_data(self, data):
        self.eeg_buffer.append(data)
        if len(self.eeg_buffer) > 100:
            self.eeg_buffer = self.eeg_buffer[-100:]
            
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.fps_label.setText(f"FPS: {self.fps:.1f}")
            self.frame_count = 0
            self.last_fps_time = current_time
            
        if self.auto_update_check.isChecked():
            self._process_eeg(data)
            
    def _process_single(self):
        data = self.eeg_source.get_sample() if hasattr(self, 'eeg_source') else SimulatedEEGSource().get_sample()
        self._process_eeg(data)
        self._log("Processed single EEG sample")
        
    def _process_eeg(self, eeg_data):
        with torch.no_grad():
            eeg_tensor = torch.from_numpy(eeg_data).unsqueeze(0).to(self.device)
            
            shape_val = self.shape_slider.value() / 100.0
            self.shape_value_label.setText(f"{self.shape_slider.value()}")
            shape_params = torch.zeros(1, 100).to(self.device)
            shape_params[0, 0] = shape_val
            
            vertices, faces, exp_params, pose_params = self.model(eeg_tensor, shape_params)
            
            exp_scale = self.exp_scale_slider.value() / 100.0
            
            new_vertices = vertices[0].cpu().numpy()
            
            if self.smooth_check.isChecked() and self.current_vertices is not None:
                alpha = 0.3
                new_vertices = alpha * new_vertices + (1 - alpha) * self.current_vertices
                
            self.current_vertices = new_vertices
            self.current_faces = faces.cpu().numpy()
            self.current_exp_params = exp_params[0].cpu().numpy()
            self.current_pose_params = pose_params[0].cpu().numpy()
            
        self._update_mesh()
        
    def _on_emotion_changed(self, emotion):
        if isinstance(self.eeg_source, SimulatedEEGSource):
            self.eeg_source.set_emotion(emotion, self.intensity_slider.value() / 100.0)
        self._log(f"Changed emotion to: {emotion}")
        
    def _on_intensity_changed(self, value):
        self.intensity_label.setText(f"{value}%")
        if isinstance(self.eeg_source, SimulatedEEGSource):
            self.eeg_source.set_emotion(self.emotion_combo.currentText(), value / 100.0)
        
    def _on_params_changed(self):
        if self.current_vertices is not None and hasattr(self, 'current_exp_params'):
            self._process_single()
            
    def _export_obj(self):
        if self.current_vertices is None:
            self._log("No mesh to export")
            return
            
        filename, _ = QFileDialog.getSaveFileName(self, "Export OBJ", "", "OBJ Files (*.obj)")
        if filename:
            with open(filename, 'w') as f:
                for v in self.current_vertices:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                for face in self.current_faces:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            self._log(f"Exported mesh to: {filename}")
            
    def _export_ply(self):
        if self.current_vertices is None:
            self._log("No mesh to export")
            return
            
        filename, _ = QFileDialog.getSaveFileName(self, "Export PLY", "", "PLY Files (*.ply)")
        if filename:
            n_verts = len(self.current_vertices)
            n_faces = len(self.current_faces)
            
            with open(filename, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {n_verts}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write(f"element face {n_faces}\n")
                f.write("property list uchar int vertex_indices\n")
                f.write("end_header\n")
                
                for v in self.current_vertices:
                    f.write(f"{v[0]} {v[1]} {v[2]}\n")
                for face in self.current_faces:
                    f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
                    
            self._log(f"Exported PLY to: {filename}")
            
    def _export_params(self):
        if not hasattr(self, 'current_exp_params') or self.current_exp_params is None:
            self._log("No parameters to export")
            return
            
        filename, _ = QFileDialog.getSaveFileName(self, "Export Parameters", "", "NumPy Files (*.npz)")
        if filename:
            np.savez(filename, 
                     expression=self.current_exp_params,
                     pose=self.current_pose_params,
                     vertices=self.current_vertices,
                     faces=self.current_faces)
            self._log(f"Exported parameters to: {filename}")
            
    def _save_model(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "PyTorch Files (*.pt *.pth)")
        if filename:
            torch.save(self.model.state_dict(), filename)
            self._log(f"Saved model to: {filename}")
            
    def _load_model(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "PyTorch Files (*.pt *.pth)")
        if filename:
            try:
                self.model.load_state_dict(torch.load(filename, map_location=self.device))
                self.model.eval()
                self._log(f"Loaded model from: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
                
    def _take_screenshot(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Screenshot", "", "PNG Files (*.png)")
        if filename:
            self.plotter_widget.screenshot(filename)
            self._log(f"Screenshot saved to: {filename}")
            
    def _log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.statusBar.showMessage(message, 3000)
        
    def closeEvent(self, event):
        if self.eeg_thread:
            self.eeg_thread.stop()
            self.eeg_thread.wait()
        self.plotter_widget.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    font = QFont()
    font.setPointSize(10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
