# Persian License Plate Detection System

## 🚗 End-to-End Car License Plate Detection and Recognition

A comprehensive real-time license plate detection and recognition system specifically designed for Persian (Iranian) license plates with database integration and native Persian text display.

## ✨ Features

### 🔍 **License Plate Detection (LPD)**
- **YOLO Model**: Uses trained `yolo12n_trained.pt` for accurate license plate detection
- **Real-time Processing**: Processes webcam feed with 2-second detection intervals
- **Multi-index Support**: Automatically tries different webcam indices (0, 1, 2)

### 📝 **License Plate Recognition (LPR)**
- **CNN Model**: EfficientNet-b0 based model (`PLPR-CNN.pth`) for text extraction
- **Persian Character Support**: Recognizes Persian letters (آ, ب, پ, ت, ث, ج, چ, ح, خ, د, ذ, ر, ز, ژ, س, ش, ص, ض, ط, ظ, ع, غ, ف, ق, ک, گ, ل, م, ن, و, ه, ی)
- **Digit Recognition**: Processes 7-digit sequences with Persian letter integration

### 🗄️ **Database Integration**
- **CSV Database**: Stores license plate information with owner details
- **Smart Matching**: 
  - Exact match lookup
  - Fuzzy matching (70% similarity threshold)
  - Persian letter variation handling (ب ↔ ل)
- **Owner Information**: Displays owner name, vehicle model, color, registration date, phone

### 🌐 **Persian Text Display**
- **Native Rendering**: Uses `arabic-reshaper` and `python-bidi` for proper Persian text
- **Font Management**: Automatic font detection with fallback options
- **Professional Overlay**: Green rectangles for found plates, red for unknown plates

### 🎥 **User Interface**
- **Webcam Integration**: Real-time video processing with mirror effect
- **Visual Feedback**: 
  - Corner markers for better visibility
  - Status indicators (DETECTION READY, PROCESSING)
  - Frame counter and statistics
- **Controls**: 
  - Press 'q' to quit
  - Press 's' to save current frame

## 🛠️ Technical Specifications

### **Models Used**
- **LPD Model**: `yolo12n_trained.pt` (YOLO v12 nano)
- **LPR Model**: `PLPR-CNN.pth` (EfficientNet-b0 backbone)

### **System Requirements**
- **Python**: 3.7+
- **RAM**: 4GB+ (8GB recommended)
- **GPU**: CUDA-compatible (optional, CPU fallback available)
- **Webcam**: Any standard USB webcam
- **OS**: Windows 10/11 (batch files included)

## 🚀 Quick Start

### **Option 1: Professional Launch (Recommended)**
1. Double-click `Run_Persian_Webcam.bat`
2. System automatically checks dependencies
3. Launches with full feature set

### **Option 2: Quick Start**
1. Double-click `Quick_Start.bat`
2. Direct launch for daily use

### **Option 3: Manual Launch**
```bash
python webcam_version_persian_final.py
```

## 📋 Usage Instructions

### **Main Menu Options**
1. **Run webcam detection** - Start real-time license plate detection
2. **Test single image** - Process a single image file
3. **Test detection drawing** - Verify overlay system with mock data
4. **Exit** - Close the application

### **Webcam Controls**
- **Detection**: Automatic every 2 seconds
- **Quit**: Press 'q' or 'Q'
- **Save Frame**: Press 's' or 'S'
- **Visual Indicators**:
  - 🟢 Green circle = Detection ready
  - 🟠 Orange circle = Processing

### **Database Format**
The system expects a CSV file (`license_plate_database.csv`) with columns:
- `plate_number`: License plate number (e.g., "12ب36511")
- `owner_name`: Owner's name in Persian
- `vehicle_model`: Vehicle model in Persian
- `vehicle_color`: Vehicle color in Persian
- `registration_date`: Registration date
- `phone_number`: Contact phone number

## 🔧 Installation

### **Automatic Installation (Recommended)**
The batch file automatically installs required packages:
```bash
pip install arabic-reshaper python-bidi
```

### **Manual Installation**
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
End-to-End-Car-Licence-Plate-Detection-and-Recognition/
├── webcam_version_persian_final.py           # Main application
├── Run_Persian_Webcam.bat                    # Professional launcher
├── Quick_Start.bat                           # Quick launcher
├── license_plate_database.csv                # License plate database
├── yolo12n_trained.pt                       # YOLO detection model
├── PLPR-CNN.pth                             # CNN recognition model
├── requirements.txt                          # Python dependencies
├── README.md                                 # This documentation
└── [other project files...]
```

## 🎯 Key Features Summary

| Feature | Description | Status |
|---------|-------------|---------|
| **Real-time Detection** | Live webcam processing | ✅ Working |
| **Persian Text Recognition** | Character and digit extraction | ✅ Working |
| **Database Lookup** | Owner information retrieval | ✅ Working |
| **Native Persian Display** | Proper text rendering | ✅ Working |
| **Professional UI** | Visual overlays and indicators | ✅ Working |
| **Error Handling** | Graceful fallbacks and recovery | ✅ Working |
| **Windows Integration** | Batch files included | ✅ Working |

## 🐛 Troubleshooting

### **Common Issues**

1. **"No module named 'arabic_reshaper'"**
   - Solution: Run the professional batch file to auto-install packages

2. **Webcam not opening**
   - Solution: Check webcam permissions and try different indices

3. **Persian text showing as ???**
   - Solution: Ensure `arabic-reshaper` and `python-bidi` are installed

4. **Model files not found**
   - Solution: Verify `yolo12n_trained.pt` and `PLPR-CNN.pth` are in the project directory

### **Performance Tips**
- Use GPU if available for faster processing
- Close other applications to free up memory
- Ensure good lighting for better detection accuracy

## 🤝 Contributing

This is a final project demonstrating:
- End-to-end machine learning pipeline
- Real-time computer vision applications
- Persian language processing
- Database integration
- Professional user interface design

## 📄 License

This project is created for educational and demonstration purposes.

## 🎉 Acknowledgments

- **YOLO Framework** for object detection
- **PyTorch** for deep learning infrastructure
- **OpenCV** for computer vision capabilities
- **Persian Language Support** libraries for native text rendering

---

**Project Status**: ✅ **COMPLETE & FUNCTIONAL**

*Ready for demonstration and deployment!* 🚀
