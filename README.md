# Nova AIGC

A powerful AI-powered web application for generating and manipulating images and videos using natural language prompts. This application provides an intuitive interface for various AI generation tasks including text-to-image, image-to-image (outpainting), text-to-video, and image-to-video conversions.

## Features

- **Text to Image Generation**: Create images from textual descriptions
- **Image to Image (Outpainting)**: Extend or modify existing images
- **Image Variation**: Generate variations of existing images
- **Text to Video Generation**: Generate videos from text descriptions
- **Image to Video Generation**: Transform static images into videos
- **Prompt Optimization**: Automatic enhancement of user prompts for better results
- **User-Friendly Interface**: Clean, intuitive Gradio-based web interface

## Prerequisites

- Python 3.8 or higher
- AWS credentials configured (for S3 access)
- Sufficient disk space for generated media files

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd nova-aigc
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment configuration:
```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your configurations
```

## Environment Configuration

The application uses environment variables for configuration. These are stored in a `.env` file:

### AWS Configuration
```
AWS_REGION=us-east-1                    # AWS region for services
AWS_S3_BUCKET=your_bucket_name          # S3 bucket for video storage
AWS_S3_PREFIX=nova-demo                 # Prefix for S3 objects
```

### Storage Paths
```
STORAGE_DIR=output                                  # Base directory for all outputs
IMAGE_OUTPUT_DIR=${STORAGE_DIR}/generated_image     # Generated images directory
VIDEO_OUTPUT_DIR=${STORAGE_DIR}/generated_videos    # Generated videos directory
TEXT2VIDEO_OUTPUT_DIR=${STORAGE_DIR}/text2video     # Text-to-video outputs
IMAGE2VIDEO_OUTPUT_DIR=${STORAGE_DIR}/image2video   # Image-to-video outputs
```

### Video Generation Parameters
```
VIDEO_DEFAULT_DURATION=6        # Default video duration in seconds
VIDEO_DEFAULT_FPS=24           # Default frames per second
VIDEO_DEFAULT_DIMENSION=1280x720 # Default video resolution
VIDEO_DEFAULT_SEED=0           # Default seed for reproducibility

VIDEO_MIN_DURATION=1           # Minimum allowed duration
VIDEO_MAX_DURATION=30          # Maximum allowed duration
VIDEO_MIN_FPS=1               # Minimum allowed FPS
VIDEO_MAX_FPS=60              # Maximum allowed FPS

VIDEO_DIMENSIONS=1280x720,1024x576,1920x1080  # Supported video dimensions
```

### Job Management
```
MAX_JOBS_TO_KEEP=50           # Maximum number of jobs to keep in history
MAX_JOB_AGE_DAYS=7           # Maximum age of jobs before cleanup
JOB_REFRESH_INTERVAL=5        # Job status refresh interval in seconds
```

### Application Settings
```
DEBUG=false                    # Enable/disable debug mode
```

## Project Structure

```
nova-aigc/
├── app.py                  # Main application file with Gradio interface
├── image_generator.py      # Image generation functionality
├── video_generator.py      # Video generation functionality
├── prompt_optimizer.py     # Prompt optimization logic
├── requirements.txt        # Project dependencies
├── .env.example           # Example environment configuration
├── .env                   # Environment variables (create from .env.example)
└── output/                # Generated media output directory
    ├── generated_videos/
    ├── generated_image/
    ├── image2video/
    └── gif/
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:7860)

3. Choose the desired generation mode:
   - **Text to Image**: Enter a prompt to generate an image
   - **Image to Image**: Upload an image and describe desired modifications
   - **Image Variation**: Upload an image and generate variations
   - **Text to Video**: Enter a prompt to generate a video
   - **Image to Video**: Upload an image and describe the desired transformation

4. For each mode:
   1. Enter your creative prompt
   2. (Optional) Use the "Optimize Prompt" feature for better results
   3. Click the respective generation button
   4. Wait for the generation process to complete
   5. View and download your generated media

## Tips for Best Results

1. **Prompt Writing**:
   - Be specific in your descriptions
   - Include details about style, mood, and quality
   - Use clear and concise language

2. **Image to Image (Outpainting)**:
   - Clearly describe which areas to modify
   - Provide specific details about desired additions or changes
   - Ensure the mask prompt accurately identifies the area to preserve

3. **Video Generation**:
   - Include temporal aspects in your descriptions
   - Specify movement and transitions
   - Consider the desired duration and pace

## Output Directory Structure

The output directory structure is configurable through environment variables:

- `output/generated_videos/`: Contains generated video files
- `output/generated_image/`: Contains generated image files
- `output/image2video/`: Contains image-to-video conversion results
- `output/gif/`: Contains generated GIF files

## Dependencies

- gradio>=4.0.0: Web interface framework
- boto3>=1.34.0: AWS SDK for Python
- python-dotenv>=1.0.0: Environment variable management
- Pillow>=10.0.0: Image processing
- moviepy>=1.0.3: Video processing
- Additional dependencies listed in requirements.txt

## Error Handling

The application includes comprehensive error handling and logging:
- All operations are logged with timestamps
- Detailed error messages are provided for troubleshooting
- Progress indicators for long-running operations

## Notes

- Ensure sufficient disk space for generated media files
- Generated files are stored in the configured output directories
- AWS credentials must be properly configured in .env for S3 access
- The application uses a local file system for temporary storage
- All configuration can be customized through environment variables
