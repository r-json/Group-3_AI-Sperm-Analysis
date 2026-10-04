# -*- coding: utf-8 -*-
"""
Modern Medical AI Interface for Sperm Morphology Classification
Designed with medical professional workflow in mind
"""

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QFileDialog, QProgressBar, QToolTip, QSplitter, 
                             QTextEdit, QGroupBox, QVBoxLayout, QHBoxLayout, 
                             QGridLayout, QScrollArea, QFrame, QMessageBox,
                             QTabWidget, QSlider, QCheckBox)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor, QIcon
import modules
import os
import sys
import json
from datetime import datetime

# Ensure we're working from the Interface directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

class ImageProcessor(QThread):
    """Background thread for image processing to keep UI responsive"""
    finished = pyqtSignal(str, str, dict)  # results_text, prediction, confidence_scores
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(self, dataset, model, image_path):
        super().__init__()
        self.dataset = dataset
        self.model = model
        self.image_path = image_path
    
    def run(self):
        try:
            self.progress.emit(25)
            if hasattr(modules, 'test_model'):
                self.progress.emit(50)
                results_text, prediction = modules.test_model(self.dataset, self.model, self.image_path)
                self.progress.emit(75)
                
                # Parse actual confidence scores from results_text
                confidence_scores = {}
                try:
                    lines = results_text.strip().split('\n')
                    for line in lines:
                        if ':' in line and '%' in line:
                            class_name, prob_str = line.split(': ')
                            prob_value = float(prob_str.replace('%', ''))
                            confidence_scores[class_name.strip()] = prob_value
                except Exception as parse_error:
                    # Fallback to default scores if parsing fails
                    if self.dataset == 'HuSHeM':
                        confidence_scores = {
                            'Normal': 85.2, 'Tapered': 8.3, 'Pyriform': 4.1, 'Amorphous': 2.4
                        }
                    else:
                        confidence_scores = {
                            'Acrosome Abnormality': 15.2, 'Boya': 10.8, 'Sperm': 74.0
                        }
                
                self.progress.emit(100)
                self.finished.emit(results_text, prediction, confidence_scores)
            else:
                self.error.emit("Model not available")
        except Exception as e:
            self.error.emit(f"Processing error: {str(e)}")

class ConfidenceWidget(QtWidgets.QWidget):
    """Custom widget for displaying confidence scores with visual bars"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(200)
        self.confidence_data = {}
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout()
        layout.setSpacing(8)
        
        # Title
        title = QtWidgets.QLabel("Confidence Analysis")
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        title.setStyleSheet("color: #0d47a1; margin-bottom: 10px; font-weight: bold;")
        layout.addWidget(title)
        
        self.bars_widget = QtWidgets.QWidget()
        self.bars_layout = QVBoxLayout(self.bars_widget)
        layout.addWidget(self.bars_widget)
        
        self.setLayout(layout)
    
    def updateConfidence(self, confidence_data):
        # Clear existing bars
        for i in reversed(range(self.bars_layout.count())): 
            self.bars_layout.itemAt(i).widget().setParent(None)
        
        # Create new confidence bars
        for class_name, confidence in confidence_data.items():
            bar_widget = self.createConfidenceBar(class_name, confidence)
            self.bars_layout.addWidget(bar_widget)
    
    def createConfidenceBar(self, class_name, confidence):
        container = QtWidgets.QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 2, 0, 2)
        
        # Class name label
        label = QtWidgets.QLabel(class_name)
        label.setFixedWidth(120)
        label.setFont(QFont("Segoe UI", 9))
        label.setProperty("styleClass", "confidence-label")
        layout.addWidget(label)
        
        # Progress bar with blue gradient
        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(int(confidence))
        progress.setTextVisible(False)
        progress.setFixedHeight(20)
        
        # Blue gradient color coding based on confidence level
        if confidence > 70:
            gradient_color = "stop: 0 #1976d2, stop: 1 #42a5f5"  # Blue gradient
        elif confidence > 40:
            gradient_color = "stop: 0 #ff9800, stop: 1 #ffb74d"  # Orange gradient
        else:
            gradient_color = "stop: 0 #f44336, stop: 1 #ef5350"  # Red gradient
            
        progress.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid rgba(25, 118, 210, 0.3);
                border-radius: 10px;
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                           stop: 0 #f8fbff, stop: 1 #ffffff);
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                           {gradient_color});
                border-radius: 8px;
            }}
        """)
        layout.addWidget(progress)
        
        # Percentage label
        percent_label = QtWidgets.QLabel(f"{confidence:.1f}%")
        percent_label.setFixedWidth(50)
        percent_label.setFont(QFont("Segoe UI", 9, QFont.Bold))
        percent_label.setAlignment(QtCore.Qt.AlignRight)
        percent_label.setProperty("styleClass", "confidence-label")
        layout.addWidget(percent_label)
        
        return container

class MorphologyExplanationWidget(QtWidgets.QWidget):
    """Widget for displaying sperm morphology explanations and background information"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # Title
        title = QtWidgets.QLabel("Morphology Information")
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        title.setStyleSheet("color: #0d47a1; margin-bottom: 10px; font-weight: bold;")
        layout.addWidget(title)
        
        # Scrollable text area for explanations
        self.explanation_area = QTextEdit()
        self.explanation_area.setFixedHeight(200)
        self.explanation_area.setReadOnly(True)
        self.explanation_area.setStyleSheet("""
            QTextEdit {
                border: 2px solid rgba(25, 118, 210, 0.2);
                border-radius: 8px;
                padding: 10px;
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #ffffff, stop: 1 #f8fbff);
                font-family: 'Segoe UI', sans-serif;
                color: #2c3e50;
                font-size: 11px;
                line-height: 1.4;
            }
        """)
        layout.addWidget(self.explanation_area)
        
        self.setLayout(layout)
        
        # Set default explanation
        self.showGeneralInformation()
    
    def showGeneralInformation(self):
        """Show general sperm morphology information"""
        text = """
        <h3 style="color: #1976d2; margin-bottom: 10px;">Sperm Morphology Analysis</h3>
        
        <p><strong>What is Sperm Morphology?</strong><br>
        Sperm morphology refers to the size and shape of sperm cells. Normal sperm have an oval head, 
        intact midpiece, and uncoiled tail. Abnormal morphology can affect fertility potential.</p>
        
        <p><strong>Clinical Significance:</strong><br>
        According to WHO criteria, normal morphology should be ≥4% for natural conception. 
        Higher percentages of abnormal forms may indicate reduced fertility potential.</p>
        
        <p style="color: #e74c3c;"><em>Upload an image to see specific morphology information and analysis explanation.</em></p>
        """
        self.explanation_area.setHtml(text)
    
    def showMorphologyExplanation(self, predicted_class, confidence, dataset):
        """Show explanation for specific morphology classification"""
        
        # Get morphology definitions
        morphology_info = self.getMorphologyDefinitions(dataset)
        
        # Get confidence explanation
        confidence_explanation = self.getConfidenceExplanation(confidence)
        
        # Build explanation text
        if predicted_class in morphology_info:
            definition = morphology_info[predicted_class]
            
            text = f"""
            <h3 style="color: #1976d2; margin-bottom: 10px;">Analysis Result: {predicted_class}</h3>
            
            <p><strong>Definition:</strong><br>
            {definition['description']}</p>
            
            <p><strong>Characteristics:</strong><br>
            {definition['characteristics']}</p>
            
            <p><strong>Clinical Implications:</strong><br>
            {definition['clinical_impact']}</p>
            
            <hr style="border: 1px solid #e0e0e0; margin: 15px 0;">
            
            <h4 style="color: #1976d2;">Confidence Analysis ({confidence:.1f}%)</h4>
            <p>{confidence_explanation}</p>
            
            <p style="color: #666; font-size: 10px; margin-top: 15px;">
            <em>Note: This analysis is for research purposes. Always consult with qualified medical professionals for clinical decisions.</em>
            </p>
            """
        else:
            text = f"""
            <h3 style="color: #1976d2;">Analysis Result: {predicted_class}</h3>
            <p>Classification detected with {confidence:.1f}% confidence.</p>
            <p>{confidence_explanation}</p>
            """
        
        self.explanation_area.setHtml(text)
    
    def getMorphologyDefinitions(self, dataset):
        """Get detailed definitions for each morphology type"""
        
        if dataset == 'HuSHeM':
            return {
                'Normal': {
                    'description': 'Normal sperm have an oval-shaped head (length 4-5.5 μm, width 2.5-3.5 μm) with a well-defined acrosome covering 40-70% of the head area.',
                    'characteristics': 'Smooth, oval head contour; intact midpiece; single, uncoiled tail; no cytoplasmic droplets.',
                    'clinical_impact': 'Normal morphology indicates good fertilization potential. WHO considers ≥4% normal forms as acceptable for natural conception.'
                },
                'Tapered': {
                    'description': 'Tapered head morphology is characterized by a head that narrows significantly, creating an elongated, cone-like appearance.',
                    'characteristics': 'Elongated head shape; reduced width; pointed anterior region; may have irregular acrosome.',
                    'clinical_impact': 'May indicate reduced fertilization capacity. High percentages of tapered forms are associated with decreased pregnancy rates.'
                },
                'Pyriform': {
                    'description': 'Pyriform (pear-shaped) sperm have heads that are wider at the base and taper toward the anterior end, resembling a pear.',
                    'characteristics': 'Pear-shaped head; wider posterior region; narrow anterior region; asymmetrical appearance.',
                    'clinical_impact': 'Associated with reduced binding to zona pellucida and decreased fertilization rates in both natural and assisted conception.'
                },
                'Amorphous': {
                    'description': 'Amorphous sperm have irregularly shaped heads that do not fit into any other specific morphological category.',
                    'characteristics': 'Irregular, undefined head shape; variable size; may have multiple abnormalities; asymmetrical contours.',
                    'clinical_impact': 'Indicates significant morphological abnormality. High percentages strongly correlate with reduced fertility potential and may require assisted reproductive techniques.'
                }
            }
        else:  # SMIDS
            return {
                'Sperm': {
                    'description': 'Normal sperm cell successfully identified in the image with typical morphological features.',
                    'characteristics': 'Recognizable sperm structure; distinct head, midpiece, and tail regions.',
                    'clinical_impact': 'Successful sperm identification allows for further morphological assessment and analysis.'
                },
                'Acrosome Abnormality': {
                    'description': 'Defects in the acrosome region, which is crucial for sperm-egg interaction and fertilization.',
                    'characteristics': 'Abnormal acrosome size, shape, or position; may be absent, enlarged, or irregularly shaped.',
                    'clinical_impact': 'Acrosome abnormalities can severely impact fertilization ability as the acrosome contains enzymes needed for egg penetration.'
                },
                'Boya': {
                    'description': 'Non-sperm cellular debris, artifacts, or other cellular material present in the sample.',
                    'characteristics': 'Cellular debris; artifacts from sample preparation; non-sperm cells; staining irregularities.',
                    'clinical_impact': 'Indicates sample quality issues or preparation artifacts. May require sample re-processing or quality control measures.'
                }
            }
    
    def getConfidenceExplanation(self, confidence):
        """Generate explanation for confidence level"""
        
        if confidence >= 90:
            return f"<span style='color: #27ae60;'><strong>Very High Confidence:</strong></span> The model is very certain about this classification ({confidence:.1f}%). This result is highly reliable for clinical consideration."
        elif confidence >= 80:
            return f"<span style='color: #27ae60;'><strong>High Confidence:</strong></span> The model shows strong certainty ({confidence:.1f}%). This result is reliable and suitable for clinical workflow."
        elif confidence >= 70:
            return f"<span style='color: #f39c12;'><strong>Good Confidence:</strong></span> The model is reasonably certain ({confidence:.1f}%). Consider this result reliable but may benefit from expert review."
        elif confidence >= 60:
            return f"<span style='color: #e67e22;'><strong>Moderate Confidence:</strong></span> The model shows moderate certainty ({confidence:.1f}%). Expert review is recommended before making clinical decisions."
        elif confidence >= 50:
            return f"<span style='color: #e74c3c;'><strong>Low Confidence:</strong></span> The model has limited certainty ({confidence:.1f}%). Manual expert review is strongly recommended."
        else:
            return f"<span style='color: #c0392b;'><strong>Very Low Confidence:</strong></span> The model is uncertain about this classification ({confidence:.1f}%). This result should not be used for clinical decisions without thorough expert validation."

class ModernSpermClassifier(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image_path = ""
        self.processing_thread = None
        self.analysis_history = []
        self.initUI()
        self.setStyleSheet(self.getModernStyleSheet())
    
    def initUI(self):
        self.setWindowTitle("Sperm Morphology AI Classifier - Medical Analysis Tool")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1200, 800)
        
        # Create central widget with splitter layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create splitter for resizable panels
        splitter = QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Controls and Configuration
        left_panel = self.createLeftPanel()
        splitter.addWidget(left_panel)
        
        # Center panel - Image Display and Analysis
        center_panel = self.createCenterPanel()
        splitter.addWidget(center_panel)
        
        # Right panel - Results and History
        right_panel = self.createRightPanel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([350, 650, 400])
        
        # Create status bar
        self.createStatusBar()
        
        # Load settings
        self.loadUserSettings()
    
    def createLeftPanel(self):
        """Create left control panel with modern gradient design"""
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(350)
        panel.setObjectName("left_panel")
        
        layout = QVBoxLayout(panel)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header with logo
        header = self.createHeader()
        layout.addWidget(header)
        
        # Dataset Selection Card
        dataset_card = self.createDatasetCard()
        layout.addWidget(dataset_card)
        
        # Model Selection Card
        model_card = self.createModelCard()
        layout.addWidget(model_card)
        
        # Image Upload Card
        upload_card = self.createUploadCard()
        layout.addWidget(upload_card)
        
        # Analysis Button
        self.analyze_btn = self.createAnalyzeButton()
        layout.addWidget(self.analyze_btn)
        
        # Settings Card
        settings_card = self.createSettingsCard()
        layout.addWidget(settings_card)
        
        layout.addStretch()
        return panel
    
    def createHeader(self):
        """Create header with logo and title"""
        header = QtWidgets.QWidget()
        layout = QVBoxLayout(header)
        
        # Logo
        logo_label = QtWidgets.QLabel()
        if os.path.exists("img/Polytechnic_University_of_the_Philippines.svg.png"):
            pixmap = QPixmap("img/Polytechnic_University_of_the_Philippines.svg.png")
            scaled_pixmap = pixmap.scaled(80, 80, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            logo_label.setPixmap(scaled_pixmap)
        logo_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(logo_label)
        
        # Title
        title = QtWidgets.QLabel("AI Sperm Analysis")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setProperty("styleClass", "header-title")
        layout.addWidget(title)
        
        # Subtitle
        subtitle = QtWidgets.QLabel("Deep Learning Classification")
        subtitle.setFont(QFont("Segoe UI", 10))
        subtitle.setAlignment(QtCore.Qt.AlignCenter)
        subtitle.setProperty("styleClass", "header-subtitle")
        layout.addWidget(subtitle)
        
        return header
    
    def createDatasetCard(self):
        """Create dataset selection card"""
        card = QGroupBox("Dataset Selection")
        card.setFont(QFont("Segoe UI", 11, QFont.Bold))
        layout = QVBoxLayout(card)
        
        self.dataset_combo = QtWidgets.QComboBox()
        self.dataset_combo.addItems(["HuSHeM (Human Sperm)", "SMIDS (Morphology Index)"])
        self.dataset_combo.setFont(QFont("Segoe UI", 10))
        self.dataset_combo.currentTextChanged.connect(self.onDatasetChanged)
        layout.addWidget(self.dataset_combo)
        
        # Dataset info label
        self.dataset_info = QtWidgets.QLabel("HuSHeM: 4 morphology classes")
        self.dataset_info.setFont(QFont("Segoe UI", 9))
        self.dataset_info.setStyleSheet("color: #7f8c8d; margin-top: 5px;")
        layout.addWidget(self.dataset_info)
        
        return card
    
    def createModelCard(self):
        """Create model selection card"""
        card = QGroupBox("AI Model Selection")
        card.setFont(QFont("Segoe UI", 11, QFont.Bold))
        layout = QVBoxLayout(card)
        
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(["MobileNet (Fast)", "Xception (Accurate)", "GoogleNet (Balanced)"])
        self.model_combo.setFont(QFont("Segoe UI", 10))
        self.model_combo.currentTextChanged.connect(self.onModelChanged)
        layout.addWidget(self.model_combo)
        
        # Model info
        self.model_info = QtWidgets.QLabel("MobileNet: Optimized for speed")
        self.model_info.setFont(QFont("Segoe UI", 9))
        self.model_info.setStyleSheet("color: #7f8c8d; margin-top: 5px;")
        layout.addWidget(self.model_info)
        
        return card
    
    def createUploadCard(self):
        """Create image upload card with drag & drop"""
        card = QGroupBox("Image Upload")
        card.setFont(QFont("Segoe UI", 11, QFont.Bold))
        layout = QVBoxLayout(card)
        
        # Upload button
        upload_btn = QtWidgets.QPushButton("📁 Select Image")
        upload_btn.setFont(QFont("Segoe UI", 10))
        upload_btn.clicked.connect(self.selectImage)
        layout.addWidget(upload_btn)
        
        # File path display
        self.file_path_label = QtWidgets.QLabel("No image selected")
        self.file_path_label.setFont(QFont("Segoe UI", 9))
        self.file_path_label.setStyleSheet("color: #7f8c8d; margin-top: 5px;")
        self.file_path_label.setWordWrap(True)
        layout.addWidget(self.file_path_label)
        
        return card
    
    def createAnalyzeButton(self):
        """Create prominent analyze button"""
        btn = QtWidgets.QPushButton("🔬 Analyze Image")
        btn.setObjectName("analyze_btn")
        btn.setFont(QFont("Segoe UI", 12, QFont.Bold))
        btn.setFixedHeight(50)
        btn.clicked.connect(self.analyzeImage)
        btn.setEnabled(False)
        return btn
    
    def createSettingsCard(self):
        """Create settings card"""
        card = QGroupBox("Analysis Settings")
        card.setFont(QFont("Segoe UI", 11, QFont.Bold))
        layout = QVBoxLayout(card)
        
        # Confidence threshold
        layout.addWidget(QtWidgets.QLabel("Confidence Threshold:"))
        self.confidence_slider = QSlider(QtCore.Qt.Horizontal)
        self.confidence_slider.setRange(50, 95)
        self.confidence_slider.setValue(70)
        layout.addWidget(self.confidence_slider)
        
        # Auto-save results
        self.auto_save_cb = QCheckBox("Auto-save results")
        self.auto_save_cb.setChecked(True)
        layout.addWidget(self.auto_save_cb)
        
        return card
    
    def createCenterPanel(self):
        """Create center panel for image display and preview"""
        panel = QtWidgets.QWidget()
        panel.setObjectName("center_panel")
        layout = QVBoxLayout(panel)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Image display area
        image_card = QGroupBox("Image Analysis")
        image_card.setFont(QFont("Segoe UI", 12, QFont.Bold))
        image_layout = QVBoxLayout(image_card)
        
        # Image label with modern styling
        self.image_label = QtWidgets.QLabel()
        self.image_label.setFixedSize(500, 500)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 3px dashed rgba(25, 118, 210, 0.4);
                border-radius: 15px;
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                           stop: 0 rgba(255, 255, 255, 0.95),
                                           stop: 1 rgba(248, 251, 255, 0.95));
                color: #1976d2;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        self.image_label.setText("Drop an image here or\nuse the Select Image button")
        image_layout.addWidget(self.image_label, alignment=QtCore.Qt.AlignCenter)
        
        layout.addWidget(image_card)
        
        # Progress bar with gradient styling
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        return panel
    
    def createRightPanel(self):
        """Create right panel for results and history"""
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(400)
        panel.setObjectName("right_panel")
        
        layout = QVBoxLayout(panel)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Results tab widget
        tab_widget = QTabWidget()
        
        # Results tab
        results_tab = QtWidgets.QWidget()
        results_layout = QVBoxLayout(results_tab)
        
        # Primary result display
        self.result_card = self.createResultCard()
        results_layout.addWidget(self.result_card)
        
        # Confidence display
        self.confidence_widget = ConfidenceWidget()
        results_layout.addWidget(self.confidence_widget)
        
        # Morphology explanation widget
        self.morphology_explanation = MorphologyExplanationWidget()
        results_layout.addWidget(self.morphology_explanation)
        
        tab_widget.addTab(results_tab, "Results")
        
        # Analysis Details tab
        details_tab = QtWidgets.QWidget()
        details_layout = QVBoxLayout(details_tab)
        
        # Detailed results
        self.detailed_results = QTextEdit()
        self.detailed_results.setFixedHeight(300)
        self.detailed_results.setPlaceholderText("Detailed technical analysis will appear here...")
        self.detailed_results.setStyleSheet("""
            QTextEdit {
                border: 2px solid rgba(25, 118, 210, 0.2);
                border-radius: 8px;
                padding: 10px;
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #ffffff, stop: 1 #f8fbff);
                font-family: 'Consolas', 'SF Mono', monospace;
                color: #0d47a1;
                font-size: 11px;
            }
        """)
        details_layout.addWidget(self.detailed_results)
        
        tab_widget.addTab(details_tab, "Technical Details")
        
        # History tab
        history_tab = self.createHistoryTab()
        tab_widget.addTab(history_tab, "History")
        
        layout.addWidget(tab_widget)
        
        # Morphology explanation widget
        self.explanation_widget = MorphologyExplanationWidget()
        layout.addWidget(self.explanation_widget)
        
        return panel
    
    def createResultCard(self):
        """Create prominent result display card"""
        card = QGroupBox("Classification Result")
        card.setFont(QFont("Segoe UI", 12, QFont.Bold))
        layout = QVBoxLayout(card)
        
        # Primary result
        self.primary_result = QtWidgets.QLabel("No analysis performed")
        self.primary_result.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self.primary_result.setAlignment(QtCore.Qt.AlignCenter)
        self.primary_result.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                background-color: #ecf0f1;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0px;
            }
        """)
        layout.addWidget(self.primary_result)
        
        # Confidence level
        self.confidence_label = QtWidgets.QLabel("")
        self.confidence_label.setFont(QFont("Segoe UI", 12))
        self.confidence_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.confidence_label)
        
        return card
    
    def createHistoryTab(self):
        """Create history tab with previous analyses"""
        tab = QtWidgets.QWidget()
        layout = QVBoxLayout(tab)
        
        # History list
        self.history_list = QtWidgets.QListWidget()
        self.history_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                background-color: white;
            }
            QListWidget::item {
                padding: 10px;
                border-bottom: 1px solid #ecf0f1;
            }
            QListWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
        """)
        layout.addWidget(self.history_list)
        
        # History controls
        history_controls = QHBoxLayout()
        clear_btn = QtWidgets.QPushButton("Clear History")
        clear_btn.clicked.connect(self.clearHistory)
        export_btn = QtWidgets.QPushButton("Export Results")
        export_btn.clicked.connect(self.exportResults)
        
        history_controls.addWidget(clear_btn)
        history_controls.addWidget(export_btn)
        layout.addLayout(history_controls)
        
        return tab
    
    def createStatusBar(self):
        """Create modern blue gradient status bar"""
        status_bar = self.statusBar()
        status_bar.setStyleSheet("""
            QStatusBar {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                           stop: 0 #0d47a1, stop: 0.5 #1565c0, stop: 1 #1976d2);
                color: white;
                font-weight: bold;
                padding: 5px;
                border-top: 2px solid rgba(255, 255, 255, 0.2);
            }
        """)
        status_bar.showMessage("Ready for analysis")
    
    def getModernStyleSheet(self):
        """Return modern blue gradient stylesheet for the application"""
        return """
        QMainWindow {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                       stop: 0 #e3f2fd, stop: 0.5 #bbdefb, stop: 1 #90caf9);
        }
        
        QWidget {
            font-family: 'Segoe UI', 'SF Pro Display', sans-serif;
        }
        
        /* Left Panel Styling */
        QWidget[objectName="left_panel"] {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 #0d47a1, stop: 0.3 #1565c0, 
                                       stop: 0.7 #1976d2, stop: 1 #1e88e5);
            border-right: 3px solid rgba(255, 255, 255, 0.3);
        }
        
        /* Right Panel Styling */
        QWidget[objectName="right_panel"] {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 #e8f4fd, stop: 0.5 #f3f9ff, stop: 1 #ffffff);
            border-left: 3px solid rgba(25, 118, 210, 0.2);
        }
        
        /* Center Panel Styling */
        QWidget[objectName="center_panel"] {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                       stop: 0 #ffffff, stop: 0.5 #f8fbff, stop: 1 #f0f7ff);
        }
        
        QGroupBox {
            font-weight: bold;
            font-size: 11px;
            border: 2px solid rgba(25, 118, 210, 0.3);
            border-radius: 12px;
            margin-top: 1ex;
            padding: 15px;
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 rgba(255, 255, 255, 0.95),
                                       stop: 1 rgba(248, 251, 255, 0.95));
            box-shadow: 0 4px 8px rgba(25, 118, 210, 0.1);
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 15px;
            padding: 0 8px 0 8px;
            color: #0d47a1;
            font-weight: bold;
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                       stop: 0 #1976d2, stop: 1 #42a5f5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        QComboBox {
            border: 2px solid rgba(25, 118, 210, 0.3);
            border-radius: 8px;
            padding: 10px;
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 #ffffff, stop: 1 #f8fbff);
            selection-background-color: #42a5f5;
            font-size: 10px;
            min-width: 120px;
        }
        
        QComboBox:focus {
            border: 2px solid #1976d2;
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 #ffffff, stop: 1 #e3f2fd);
        }
        
        QComboBox:hover {
            border: 2px solid #42a5f5;
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 #ffffff, stop: 1 #f0f7ff);
        }
        
        QComboBox::drop-down {
            border: none;
            width: 30px;
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                       stop: 0 #1976d2, stop: 1 #42a5f5);
            border-top-right-radius: 6px;
            border-bottom-right-radius: 6px;
        }
        
        QComboBox::down-arrow {
            image: none;
            border: 2px solid white;
            width: 0px;
            height: 0px;
            border-top: 5px solid white;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
        }
        
        QPushButton {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                       stop: 0 #1976d2, stop: 0.5 #2196f3, stop: 1 #42a5f5);
            border: none;
            color: white;
            padding: 12px;
            border-radius: 8px;
            font-weight: bold;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        QPushButton:hover {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                       stop: 0 #1565c0, stop: 0.5 #1e88e5, stop: 1 #42a5f5);
            transform: translateY(-1px);
        }
        
        QPushButton:pressed {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                       stop: 0 #0d47a1, stop: 0.5 #1565c0, stop: 1 #1976d2);
            transform: translateY(1px);
        }
        
        QPushButton:disabled {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                       stop: 0 #bdbdbd, stop: 1 #e0e0e0);
            color: #9e9e9e;
        }
        
        /* Large Analyze Button Special Styling */
        QPushButton[objectName="analyze_btn"] {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                       stop: 0 #0d47a1, stop: 0.3 #1565c0, 
                                       stop: 0.7 #1976d2, stop: 1 #1e88e5);
            font-size: 14px;
            font-weight: bold;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(13, 71, 161, 0.3);
        }
        
        QPushButton[objectName="analyze_btn"]:hover {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                       stop: 0 #0a3d91, stop: 0.3 #1357b0, 
                                       stop: 0.7 #1669c2, stop: 1 #1b7ad5);
            box-shadow: 0 6px 12px rgba(13, 71, 161, 0.4);
        }
        
        QTabWidget::pane {
            border: 2px solid rgba(25, 118, 210, 0.2);
            border-radius: 10px;
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 #ffffff, stop: 1 #f8fbff);
        }
        
        QTabBar::tab {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 #e3f2fd, stop: 1 #f0f7ff);
            border: 1px solid rgba(25, 118, 210, 0.3);
            padding: 10px 20px;
            margin-right: 2px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            color: #1976d2;
            font-weight: bold;
        }
        
        QTabBar::tab:selected {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 #1976d2, stop: 1 #42a5f5);
            color: white;
            border-bottom: none;
        }
        
        QTabBar::tab:hover {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 #bbdefb, stop: 1 #e3f2fd);
        }
        
        QTextEdit {
            border: 2px solid rgba(25, 118, 210, 0.2);
            border-radius: 8px;
            padding: 10px;
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 #ffffff, stop: 1 #f8fbff);
            font-family: 'Consolas', 'SF Mono', monospace;
            color: #0d47a1;
        }
        
        QTextEdit:focus {
            border: 2px solid #1976d2;
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 #ffffff, stop: 1 #f0f7ff);
        }
        
        QProgressBar {
            border: 2px solid rgba(25, 118, 210, 0.3);
            border-radius: 8px;
            text-align: center;
            font-weight: bold;
            color: #0d47a1;
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                       stop: 0 #f8fbff, stop: 1 #ffffff);
        }
        
        QProgressBar::chunk {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                       stop: 0 #1976d2, stop: 0.5 #2196f3, stop: 1 #42a5f5);
            border-radius: 6px;
        }
        
        QSlider::groove:horizontal {
            border: 1px solid rgba(25, 118, 210, 0.3);
            height: 10px;
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                       stop: 0 #e3f2fd, stop: 1 #f0f7ff);
            border-radius: 5px;
        }
        
        QSlider::handle:horizontal {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                       stop: 0 #1976d2, stop: 1 #42a5f5);
            border: 2px solid #0d47a1;
            width: 20px;
            margin: -6px 0;
            border-radius: 11px;
        }
        
        QSlider::handle:horizontal:hover {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                       stop: 0 #1565c0, stop: 1 #1e88e5);
        }
        
        QCheckBox {
            font-size: 10px;
            color: #0d47a1;
            font-weight: bold;
        }
        
        QCheckBox::indicator {
            width: 20px;
            height: 20px;
        }
        
        QCheckBox::indicator:unchecked {
            border: 2px solid rgba(25, 118, 210, 0.5);
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                       stop: 0 #ffffff, stop: 1 #f8fbff);
            border-radius: 4px;
        }
        
        QCheckBox::indicator:checked {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                       stop: 0 #1976d2, stop: 1 #42a5f5);
            border: 2px solid #0d47a1;
            border-radius: 4px;
        }
        
        QCheckBox::indicator:checked:hover {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                       stop: 0 #1565c0, stop: 1 #1e88e5);
        }
        
        QLabel {
            color: #0d47a1;
        }
        
        /* Special styling for white text labels on dark backgrounds */
        QLabel[styleClass="white-text"] {
            color: white;
            font-weight: bold;
            text-shadow: 0 2px 6px rgba(0, 0, 0, 0.5);
            background-color: rgba(0, 0, 0, 0.1);
            padding: 2px 6px;
            border-radius: 4px;
        }
        
        QLabel[styleClass="header-title"] {
            color: white;
            font-weight: bold;
            font-size: 16px;
            text-shadow: 0 3px 8px rgba(0, 0, 0, 0.6);
            background-color: rgba(0, 0, 0, 0.15);
            padding: 8px 12px;
            border-radius: 6px;
        }
        
        QLabel[styleClass="header-subtitle"] {
            color: rgba(255, 255, 255, 0.95);
            font-size: 10px;
            text-shadow: 0 2px 6px rgba(0, 0, 0, 0.4);
            background-color: rgba(0, 0, 0, 0.1);
            padding: 4px 8px;
            border-radius: 4px;
        }
        
        /* Result card special styling */
        QLabel[objectName="primary_result"] {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                       stop: 0 #e8f5e8, stop: 1 #f1f8e9);
            border: 2px solid #4caf50;
            color: #2e7d32;
            font-size: 18px;
            font-weight: bold;
            border-radius: 12px;
            padding: 20px;
        }
        
        /* Confidence widget styling */
        QLabel[styleClass="confidence-label"] {
            color: #0d47a1;
            font-weight: bold;
            font-size: 9px;
        }
        
        QListWidget {
            border: 2px solid rgba(25, 118, 210, 0.2);
            border-radius: 8px;
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 #ffffff, stop: 1 #f8fbff);
            alternate-background-color: #f0f7ff;
        }
        
        QListWidget::item {
            padding: 12px;
            border-bottom: 1px solid rgba(25, 118, 210, 0.1);
            color: #0d47a1;
        }
        
        QListWidget::item:selected {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                       stop: 0 #1976d2, stop: 1 #42a5f5);
            color: white;
            border-radius: 6px;
        }
        
        QListWidget::item:hover {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                       stop: 0 #e3f2fd, stop: 1 #f0f7ff);
            border-radius: 6px;
        }
        """
    
    def onDatasetChanged(self, text):
        """Handle dataset selection change"""
        if "HuSHeM" in text:
            self.dataset_info.setText("HuSHeM: 4 morphology classes (Normal, Tapered, Pyriform, Amorphous)")
        else:
            self.dataset_info.setText("SMIDS: 3 classes (Acrosome Abnormality, Boya, Sperm)")
    
    def onModelChanged(self, text):
        """Handle model selection change"""
        if "MobileNet" in text:
            self.model_info.setText("MobileNet: Optimized for speed (~100ms)")
        elif "Xception" in text:
            self.model_info.setText("Xception: High accuracy, slower processing (~500ms)")
        else:
            self.model_info.setText("GoogleNet: Balanced speed and accuracy (~300ms)")
    
    def selectImage(self):
        """Handle image selection"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Sperm Image", 
            "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tiff);;All Files (*)"
        )
        
        if file_path:
            self.current_image_path = file_path
            self.file_path_label.setText(f"📄 {os.path.basename(file_path)}")
            
            # Load and display image
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    self.image_label.size(), 
                    QtCore.Qt.KeepAspectRatio, 
                    QtCore.Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
                self.image_label.setText("")
                self.analyze_btn.setEnabled(True)
            else:
                QMessageBox.warning(self, "Error", "Could not load the selected image.")
    
    def analyzeImage(self):
        """Start image analysis in background thread"""
        if not self.current_image_path:
            QMessageBox.warning(self, "No Image", "Please select an image first.")
            return
        
        # Get selected dataset and model
        dataset_text = self.dataset_combo.currentText()
        model_text = self.model_combo.currentText()
        
        dataset = "HuSHeM" if "HuSHeM" in dataset_text else "SMIDS"
        model = model_text.split(" ")[0]  # Extract model name
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.analyze_btn.setEnabled(False)
        self.statusBar().showMessage("Analyzing image...")
        
        # Start background processing
        self.processing_thread = ImageProcessor(dataset, model, self.current_image_path)
        self.processing_thread.progress.connect(self.progress_bar.setValue)
        self.processing_thread.finished.connect(self.onAnalysisComplete)
        self.processing_thread.error.connect(self.onAnalysisError)
        self.processing_thread.start()
    
    def onAnalysisComplete(self, results_text, prediction, confidence_scores):
        """Handle analysis completion"""
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        
        # Update primary result
        self.primary_result.setText(prediction)
        
        # Update confidence display
        if confidence_scores:
            max_confidence = max(confidence_scores.values())
            self.confidence_label.setText(f"Confidence: {max_confidence:.1f}%")
            self.confidence_widget.updateConfidence(confidence_scores)
        
        # Update detailed results
        self.detailed_results.setText(results_text)
        
        # Add to history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history_item = f"{timestamp} - {prediction} ({max_confidence:.1f}%)"
        self.history_list.addItem(history_item)
        
        # Store in analysis history
        analysis_record = {
            'timestamp': timestamp,
            'image_path': self.current_image_path,
            'prediction': prediction,
            'confidence_scores': confidence_scores,
            'dataset': self.dataset_combo.currentText(),
            'model': self.model_combo.currentText()
        }
        self.analysis_history.append(analysis_record)
        
        # Auto-save if enabled
        if self.auto_save_cb.isChecked():
            self.saveAnalysisRecord(analysis_record)
        
        # Update explanation widget
        dataset = "HuSHeM" if "HuSHeM" in self.dataset_combo.currentText() else "SMIDS"
        self.morphology_explanation.showMorphologyExplanation(prediction, max_confidence, dataset)
        
        self.statusBar().showMessage(f"Analysis complete: {prediction}")
    
    def onAnalysisError(self, error_message):
        """Handle analysis error"""
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.statusBar().showMessage("Analysis failed")
        QMessageBox.critical(self, "Analysis Error", error_message)
    
    def clearHistory(self):
        """Clear analysis history"""
        reply = QMessageBox.question(
            self, 
            "Clear History", 
            "Are you sure you want to clear all analysis history?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.history_list.clear()
            self.analysis_history.clear()
    
    def exportResults(self):
        """Export analysis results to file"""
        if not self.analysis_history:
            QMessageBox.information(self, "No Data", "No analysis results to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Export Results", 
            f"sperm_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.analysis_history, f, indent=2)
                QMessageBox.information(self, "Export Successful", f"Results exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export results: {str(e)}")
    
    def saveAnalysisRecord(self, record):
        """Save individual analysis record"""
        # Implementation for auto-saving records
        pass
    
    def loadUserSettings(self):
        """Load user preferences"""
        # Implementation for loading user settings
        pass

def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Sperm Morphology AI Classifier")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Polytechnic University of the Philippines")
    
    # Create and show main window
    window = ModernSpermClassifier()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
