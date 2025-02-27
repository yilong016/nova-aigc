# Nova AIGC

AI-powered application for generating and manipulating images and videos using AWS Bedrock Nova models.

## Features

### Image Generation
- Text to Image Generation
- Text to Image with Conditioning (Edge/Segmentation)
- Color Guided Image Generation
- Image Variation Generation
- Image Inpainting
- Image Outpainting
- Background Removal
- Prompt Optimization for Better Results
- Advanced Options (Quality, CFG Scale, Dimensions, etc.)

### Object Detection and Masking
- Detect specific objects in images
- Generate binary masks (black objects on white background)
- Support for both Nova Lite and Pro models
- Adjustable image processing size
- Coordinates output for detected objects
- Automatic mask file saving

### Video Generation
- Text to Video Generation
- Image to Video Transformation
- Video Prompt Optimization

### Interface
- User-friendly Web Interface
- Real-time Generation Progress
- Multiple Generation Options
- Advanced Configuration Controls

## Setup

1. Install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your AWS credentials and settings
```

## Environment Variables

Essential configurations in `.env`:

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_S3_BUCKET=your_bucket_name
AWS_S3_PREFIX=nova-demo

# Model Configuration
NOVA_MODEL_ID=us.amazon.nova-pro-v1:0
DEFAULT_MODEL=us.amazon.nova-lite-v1:0

# Storage Paths
STORAGE_DIR=output
MASK_DIR=${STORAGE_DIR}/mask
UPLOAD_DIR=${STORAGE_DIR}/uploads
```

## Usage

1. Start the web interface:
```bash
python app.py
```

2. Open http://localhost:7860 in your browser

3. Choose your generation mode:
   - Image Generation: Create and manipulate images with various options
   - Object Detection: Detect objects and create masks
   - Text to Video: Generate videos from text descriptions
   - Image to Video: Transform images into videos
   - Use "Optimize Prompt" for better results

### Object Detection and Masking

1. Select the "Object Detection" tab
2. Upload an image
3. Enter the name of the object to detect (e.g., "bottle", "car", "dog")
4. Choose the model (Nova Lite or Pro)
5. Adjust the image processing size if needed
6. Click "Detect and Create Mask"
7. View results:
   - Detection Result: Original image with detection boxes
   - Mask Result: Binary mask with black objects on white background
   - Detection Coordinates: JSON output of object locations
   - Saved Mask Path: Location of the saved mask file

## Requirements

- Python 3.12+
- AWS credentials with Bedrock access
