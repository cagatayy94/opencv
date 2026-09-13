# Real-Time Image Processor

A Python application for real-time image processing using OpenCV and a webcam. This tool allows you to dynamically toggle between different computer vision algorithms with simple keyboard shortcuts.

## Current Features
* **Normal Mode (`n`)**: Raw webcam feed with no processing.
* **Face Detection (`f`)**: Real-time face detection using Haar Cascades.

*More features (Object Detection, Edge Detection, etc.) are currently in development!*

## Prerequisites
* Python 3.10+
* A webcam

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd opencv
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script to start the application:

```bash
python main.py
```

### Controls
Once the window opens, ensure it is in focus and use the following keys to toggle modes:
* `n`: Normal view
* `f`: Face detection view
* `q`: Quit the application
