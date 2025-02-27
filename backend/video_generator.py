import json
import boto3
import base64
import os
import time
import logging
from typing import Dict, Optional
from urllib.parse import urlparse
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class NovaVideoGenerator:
    """Simple Nova Video Generator for text-to-video and image-to-video generation."""
    
    # Supported image formats
    SUPPORTED_FORMATS = {
        'jpg': 'jpeg',
        'jpeg': 'jpeg',
        'png': 'png'
    }
    
    # Required dimensions for Nova Reel
    REQUIRED_WIDTH = 1280
    REQUIRED_HEIGHT = 720
    
    def __init__(self, s3_output_bucket: Optional[str] = None):
        """Initialize the Nova Video Generator.

        Args:
            s3_output_bucket (Optional[str]): Override the S3 bucket URI from env vars
        """
        logger.info("Initializing NovaVideoGenerator...")
        
        # Use provided bucket or construct from env vars
        if s3_output_bucket is None:
            bucket = os.getenv('AWS_S3_BUCKET')
            prefix = os.getenv('AWS_S3_PREFIX', 'nova-demo')
            s3_output_bucket = f"s3://{bucket}/{prefix}/"
        
        logger.info(f"Output bucket: {s3_output_bucket}")
        
        self.bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        self.s3_client = boto3.client('s3')
        self.s3_output_bucket = s3_output_bucket
        
        # Set up output directories from env vars
        self.text2video_dir = os.getenv('TEXT2VIDEO_OUTPUT_DIR', './output/text2video')
        self.image2video_dir = os.getenv('IMAGE2VIDEO_OUTPUT_DIR', './output/image2video')
        os.makedirs(self.text2video_dir, exist_ok=True)
        os.makedirs(self.image2video_dir, exist_ok=True)
        
        logger.info("NovaVideoGenerator initialized successfully")

    def _validate_image(self, image_path: str) -> str:
        """Validate image format and return the standardized format name.

        Args:
            image_path (str): Path to the image file

        Returns:
            str: Standardized format name ('jpeg' or 'png')

        Raises:
            ValueError: If image format is not supported
        """
        logger.debug(f"Validating image: {image_path}")
        
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            raise ValueError(f"Image file not found: {image_path}")
            
        ext = image_path.lower().split('.')[-1]
        if ext not in self.SUPPORTED_FORMATS:
            logger.error(f"Unsupported image format: {ext}")
            raise ValueError(
                f"Unsupported image format: {ext}. "
                f"Supported formats are: {', '.join(set(self.SUPPORTED_FORMATS.values()))}"
            )
            
        logger.debug(f"Image validation successful: {image_path}")
        return self.SUPPORTED_FORMATS[ext]

    def generate_video(self, text: str, input_image_path: Optional[str] = None) -> Dict:
        """Generate video from text, optionally with an input image.
        If input_image_path is provided, generates image-to-video.
        If only text is provided, generates text-to-video.
        Only JPEG and PNG image formats are supported.

        Args:
            text (str): The text description for video generation
            input_image_path (Optional[str]): Path to input image (JPEG or PNG). If None, does text-to-video

        Returns:
            Dict: Response containing job information

        Raises:
            ValueError: If image format is not supported or file not found
        """
        try:
            # Log generation start
            logger.info("Starting video generation")
            logger.info(f"Input text: {text}")
            
            # Prepare base model input
            model_input = {
                "taskType": "TEXT_VIDEO",
                "textToVideoParams": {
                    "text": text
                },
                "videoGenerationConfig": {
                    "durationSeconds": int(os.getenv('VIDEO_DEFAULT_DURATION', '6')),
                    "fps": int(os.getenv('VIDEO_DEFAULT_FPS', '24')),
                    "dimension": os.getenv('VIDEO_DEFAULT_DIMENSION', '1280x720'),
                    "seed": int(os.getenv('VIDEO_DEFAULT_SEED', '0'))
                }
            }

            # Add image data if provided (image-to-video mode)
            if input_image_path:
                # Validate image format
                image_format = self._validate_image(input_image_path)
                logger.info(f"Generating image-to-video using {image_format} image: {input_image_path}")
                
                # Read and encode image
                with open(input_image_path, "rb") as f:
                    image_base64 = base64.b64encode(f.read()).decode("utf-8")
                model_input["textToVideoParams"]["images"] = [{
                    "format": image_format,
                    "source": {
                        "bytes": image_base64
                    }
                }]
            else:
                logger.info("Generating text-to-video")
            
            # Start video generation
            logger.info("Calling Bedrock API for video generation...")
            response = self.bedrock_runtime.start_async_invoke(
                modelId="amazon.nova-reel-v1:0",
                modelInput=model_input,
                outputDataConfig={
                    "s3OutputDataConfig": {
                        "s3Uri": self.s3_output_bucket
                    }
                }
            )
            logger.info(f"Generation job started successfully: {response.get('invocationArn', 'No ARN available')}")
            return response
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            raise e
        except Exception as e:
            if hasattr(e, 'response'):
                message = e.response["Error"]["Message"]
                logger.error(f"API error: {message}")
                raise Exception(f"Video generation failed: {message}")
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            raise e

    def get_job_status(self, invocation_arn: str) -> Dict:
        """Get the status of a video generation job.

        Args:
            invocation_arn (str): The ARN of the invocation to check

        Returns:
            Dict: Job status information
        """
        logger.debug(f"Checking status for job: {invocation_arn}")
        invocation = self.bedrock_runtime.get_async_invoke(invocationArn=invocation_arn)
        
        status = invocation["status"]
        result = {
            "status": status,
            "invocation_arn": invocation_arn
        }

        if status == "Completed":
            bucket_uri = invocation["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
            result["video_uri"] = f"{bucket_uri}/output.mp4"
            logger.info(f"Job completed successfully. Video available at: {result['video_uri']}")
        elif status == "Failed":
            result["failure_message"] = invocation["failureMessage"]
            logger.error(f"Job failed: {result['failure_message']}")
        else:
            logger.info(f"Job status: {status}")

        return result

    def wait_for_completion(self, invocation_arn: str, check_interval: int = 10) -> Dict:
        """Wait for a job to complete and return its final status.

        Args:
            invocation_arn (str): The ARN of the invocation to check
            check_interval (int): How often to check status in seconds

        Returns:
            Dict: Final job status
        """
        logger.info(f"Waiting for job completion: {invocation_arn}")
        while True:
            status = self.get_job_status(invocation_arn)
            if status["status"] in ["Completed", "Failed"]:
                logger.info(f"Job finished with status: {status['status']}")
                return status
            logger.info(f"Job status: {status['status']}... waiting {check_interval} seconds")
            time.sleep(check_interval)

    def download_video(self, video_uri: str, is_text_to_video: bool = True) -> str:
        """Download the generated video from S3 to a local directory.

        Args:
            video_uri (str): S3 URI of the video (e.g., s3://bucket/path/output.mp4)
            is_text_to_video (bool): Whether this is a text-to-video (True) or image-to-video (False) download

        Returns:
            str: Path to the downloaded video file

        Raises:
            ValueError: If video_uri is invalid
        """
        try:
            logger.info(f"Starting video download from {video_uri}")
            
            # Parse S3 URI
            parsed_uri = urlparse(video_uri)
            if parsed_uri.scheme != 's3':
                logger.error(f"Invalid S3 URI: {video_uri}")
                raise ValueError(f"Invalid S3 URI: {video_uri}")
            
            # Extract bucket and key
            bucket = parsed_uri.netloc
            key = parsed_uri.path.lstrip('/')
            
            # Choose output directory based on video type
            local_dir = self.text2video_dir if is_text_to_video else self.image2video_dir
            
            # Generate local file path with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename, ext = os.path.splitext(os.path.basename(key))
            local_filename = os.path.join(local_dir, f"{filename}_{timestamp}{ext}")
            
            logger.info(f"Downloading video to {local_filename}")
            
            # Download file
            self.s3_client.download_file(bucket, key, local_filename)
            
            logger.info(f"Video downloaded successfully to {local_filename}")
            return local_filename
            
        except Exception as e:
            logger.error(f"Failed to download video: {str(e)}", exc_info=True)
            raise Exception(f"Failed to download video: {str(e)}")


# Example usage:
if __name__ == "__main__":
    try:
        # Initialize generator with S3 bucket from environment variables
        generator = NovaVideoGenerator()

        # Example 1: Text-to-video generation with prompt optimization
        logger.info("\n=== Starting Text-to-Video Generation ===")
        text = "A beautiful sunset over a mountain landscape, cinematic quality"
        # Prompt optimization can be done outside if needed
        from prompt_optimizer import PromptOptimizer
        prompt_optimizer = PromptOptimizer()
        optimized_text = prompt_optimizer.optimize_prompt(text)
        text_response = generator.generate_video(optimized_text)
        logger.info(f"Job started: {json.dumps(text_response, indent=2)}")
        
        # Wait for text-to-video job completion
        if "invocationArn" in text_response:
            logger.info("\nWaiting for text-to-video job completion...")
            final_status = generator.wait_for_completion(text_response["invocationArn"])
            logger.info(f"Text-to-video final status: {json.dumps(final_status, indent=2)}")
            
            # Download the video if job completed successfully
            if final_status["status"] == "Completed":
                video_path = generator.download_video(final_status["video_uri"], is_text_to_video=True)
                logger.info(f"Video downloaded to: {video_path}")

        # Example 2: Image-to-video generation without prompt optimization
        logger.info("\n=== Starting Image-to-Video Generation ===")
        text = "First Person View Aerial, Dolly In Shot, Ultra HD, 8K resolution"
        image_path = "./data/20241204/d9ab9ef4_z.jpg"
        image_response = generator.generate_video(text, input_image_path=image_path)
        logger.info(f"Job started: {json.dumps(image_response, indent=2)}")
        
        # Wait for image-to-video job completion
        if "invocationArn" in image_response:
            logger.info("\nWaiting for image-to-video job completion...")
            final_status = generator.wait_for_completion(image_response["invocationArn"])
            logger.info(f"Image-to-video final status: {json.dumps(final_status, indent=2)}")
            
            # Download the video if job completed successfully
            if final_status["status"] == "Completed":
                video_path = generator.download_video(final_status["video_uri"], is_text_to_video=False)
                logger.info(f"Video downloaded to: {video_path}")

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
