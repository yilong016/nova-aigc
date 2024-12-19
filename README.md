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
AWS_REGION=us-east-1
AWS_S3_BUCKET=your_bucket_name
AWS_S3_PREFIX=nova-demo
# model id used for prompt optimization
NOVA_MODEL_ID=us.amazon.nova-pro-v1:0
```

## Usage

1. Start the web interface:
```bash
python app.py
```

2. Open http://localhost:7860 in your browser

3. Choose your generation mode:
   - Image Generation: Create and manipulate images with various options
   - Text to Video: Generate videos from text descriptions
   - Image to Video: Transform images into videos
   - Use "Optimize Prompt" for better results


## Requirements

- Python 3.12+
- AWS credentials with Bedrock access
