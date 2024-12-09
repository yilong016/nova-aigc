# Nova AIGC

AI-powered web application for generating and manipulating images and videos using AWS Bedrock Nova models.

## Features

- Text to Image Generation
- Image to Image (Outpainting)
- Image Variation Generation
- Text to Video Generation
- Image to Video Generation
- Automatic Prompt Optimization
- Gradio Web Interface

## Setup

1. Clone and install dependencies:
```bash
git clone <repository-url>
cd nova-aigc
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

## Configuration

Key environment variables in `.env`:

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_S3_BUCKET=your_bucket_name
AWS_S3_PREFIX=nova-demo

# Nova Configuration
NOVA_MODEL_ID=us.amazon.nova-pro-v1:0

# Storage Configuration
STORAGE_DIR=output
IMAGE_OUTPUT_DIR=${STORAGE_DIR}/generated_image
VIDEO_OUTPUT_DIR=${STORAGE_DIR}/generated_videos
TEXT2VIDEO_OUTPUT_DIR=${STORAGE_DIR}/text2video
IMAGE2VIDEO_OUTPUT_DIR=${STORAGE_DIR}/image2video

# Video Parameters
VIDEO_DEFAULT_DURATION=6
VIDEO_DEFAULT_FPS=24
VIDEO_DEFAULT_DIMENSION=1280x720
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open http://localhost:7860 in your browser

3. Select generation mode:
   - Text to Image: Generate images from text descriptions
   - Image to Image: Modify existing images
   - Image Variation: Create variations of images
   - Text to Video: Generate videos from text
   - Image to Video: Transform images into videos

4. Enter your prompt and use the "Optimize Prompt" feature for better results

## Project Structure

```
nova-aigc/
├── app.py                  # Main Gradio interface
├── image_generator.py      # Image generation
├── video_generator.py      # Video generation
├── prompt_optimizer.py     # Prompt optimization
├── nova_reel_prompts.py   # Video generation prompts
├── nova_canvas_prompts.py # Image generation prompts
└── output/                # Generated media
```

## Requirements

- Python 3.8+
- AWS credentials configured
- Dependencies listed in requirements.txt
