import json
import boto3
import base64
import os
import logging
import imghdr
from typing import Dict, Optional, Union, List
from PIL import Image
from io import BytesIO
from botocore.config import Config
from datetime import datetime
from backend.prompt_optimizer import CanvasPromptOptimizer
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

class NovaImageGenerator:
    """Nova Image Generator for all image generation tasks."""
    
    def __init__(self):
        """Initialize the Nova Image Generator."""
        logger.info("Initializing NovaImageGenerator...")
        
        self.bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name=os.getenv('AWS_REGION', 'us-east-1'),
            config=Config(read_timeout=300)
        )
        self.output_dir = os.getenv('IMAGE_OUTPUT_DIR', './output/generated_image')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.model_id = "amazon.nova-canvas-v1:0"
            
        logger.info("NovaImageGenerator initialized successfully")

    def _validate_config(self, task_type: str, config: Dict) -> Dict:
        """Validate and filter configuration based on task type."""
        valid_config = {}
        
        # Handle seed value first - if it's -1 or not provided, generate a random seed
        if "seed" in config:
            if config["seed"] < 0:
                import random
                config["seed"] = random.randint(0, 858993459)
        
        if task_type == "BACKGROUND_REMOVAL":
            # No config needed for background removal
            return valid_config
            
        elif task_type in ["INPAINTING", "OUTPAINTING"]:
            # Only these parameters are valid for inpainting/outpainting
            valid_keys = ["numberOfImages", "quality", "cfgScale", "seed"]
            for key in valid_keys:
                if key in config:
                    valid_config[key] = config[key]
                    
        elif task_type == "IMAGE_VARIATION":
            # Image variation doesn't use quality
            valid_keys = ["numberOfImages", "height", "width", "cfgScale", "seed"]
            for key in valid_keys:
                if key in config:
                    valid_config[key] = config[key]
                    
        else:  # TEXT_IMAGE and COLOR_GUIDED_GENERATION
            # All parameters are valid
            valid_keys = ["width", "height", "quality", "cfgScale", "seed", "numberOfImages"]
            for key in valid_keys:
                if key in config:
                    valid_config[key] = config[key]
        
        return valid_config

    def _invoke_model(self, body: Dict) -> bytes:
        """Invoke the Bedrock model with the given body."""
        try:
            logger.info("Calling Bedrock API for image generation...")
            response = self.bedrock_runtime.invoke_model(
                body=json.dumps(body),
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json"
            )
            
            response_body = json.loads(response.get("body").read())
            
            if "error" in response_body and response_body["error"]:
                error_msg = response_body["error"]
                logger.error(f"Image generation failed: {error_msg}")
                raise Exception(f"Image generation failed: {error_msg}")
            
            base64_images = response_body.get("images", [])
            return [base64.b64decode(img.encode('ascii')) for img in base64_images]
            
        except Exception as e:
            logger.error(f"Model invocation failed: {str(e)}", exc_info=True)
            raise

    def _save_images(self, image_bytes_list: List[bytes], task_type: str = "generated") -> List[str]:
        """Save multiple image bytes to files with timestamps."""
        output_paths = []
        base_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for idx, image_bytes in enumerate(image_bytes_list):
            output_path = os.path.join(self.output_dir, f"{task_type.lower()}_image_{base_timestamp}_{idx+1}.png")
            image = Image.open(BytesIO(image_bytes))
            image.save(output_path, format='PNG')
            output_paths.append(output_path)
            logger.info(f"Image saved to: {output_path}")
            
        return output_paths

    def _convert_webp_to_png(self, image_path: str) -> str:
        """Convert WebP image to PNG format.
        
        Args:
            image_path (str): Path to the WebP image
            
        Returns:
            str: Path to the converted PNG image
        """
        output_path = os.path.splitext(image_path)[0] + '.png'
        with Image.open(image_path) as img:
            # Convert to RGB if image is in RGBA mode
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            img.save(output_path, 'PNG', quality=100)
        return output_path

    def _verify_image_format(self, image_path: str) -> str:
        """Verify that the image is in JPEG or PNG format, convert if WebP.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Path to the verified/converted image
        """
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")

        # First try to detect using imghdr
        actual_type = imghdr.what(image_path)
        logger.info(f"Detected image type: {actual_type}")

        # If imghdr fails or returns None, try using PIL
        if not actual_type:
            try:
                with Image.open(image_path) as img:
                    actual_type = img.format.lower()
                    logger.info(f"PIL detected image type: {actual_type}")
            except Exception as e:
                raise ValueError(f"Could not determine image format: {str(e)}")

        if actual_type == 'webp':
            logger.info("Converting WebP image to PNG format")
            return self._convert_webp_to_png(image_path)
        elif actual_type not in ['jpeg', 'png']:
            raise ValueError(f"Invalid image format. Expected JPEG, PNG, or WebP, but got: {actual_type}")
        
        return image_path

    def _encode_image(self, image_path: str) -> str:
        """Read and base64 encode an image file."""
        image_path = self._verify_image_format(image_path)
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf8')

    def generate(
        self,
        task_type: str,
        params: Dict,
        config: Dict
    ) -> str:
        """Unified method for all image generation tasks.
        
        Args:
            task_type (str): Type of generation task (TEXT_IMAGE, COLOR_GUIDED_GENERATION, etc.)
            params (Dict): Task-specific parameters
            config (Dict): Image generation configuration
            
        Returns:
            str: Path to the generated image
        """
        try:
            logger.info(f"Starting {task_type} generation")
            logger.info(f"Parameters: {params}")
            logger.info(f"Configuration: {config}")

            # Validate and filter configuration based on task type
            valid_config = self._validate_config(task_type, config)
            logger.info(f"Validated configuration: {valid_config}")

            # Handle image encoding for tasks that require images
            if task_type != "TEXT_IMAGE" or "conditionImage" in params:
                if "image" in params and params["image"]:
                    params["image"] = self._encode_image(params["image"])
                if "images" in params and params["images"]:
                    params["images"] = [self._encode_image(img) for img in params["images"] if img]
                if "referenceImage" in params and params["referenceImage"]:
                    params["referenceImage"] = self._encode_image(params["referenceImage"])
                if "conditionImage" in params and params["conditionImage"]:
                    params["conditionImage"] = self._encode_image(params["conditionImage"])
                if "maskImage" in params and params["maskImage"]:
                    params["maskImage"] = self._encode_image(params["maskImage"])

            # Prepare request body
            body = {
                "taskType": task_type,
                "imageGenerationConfig": valid_config
            }

            # Add task-specific parameters
            if task_type == "TEXT_IMAGE":
                if "conditionImage" in params:
                    body["textToImageParams"] = {
                        "text": params["text"],
                        "negativeText": params.get("negativeText", ""),
                        "conditionImage": params["conditionImage"],
                        "controlMode": params["controlMode"],
                        "controlStrength": params["controlStrength"]
                    }
                else:
                    body["textToImageParams"] = {
                        "text": params["text"],
                        "negativeText": params.get("negativeText", "")
                    }
            elif task_type == "COLOR_GUIDED_GENERATION":
                body["colorGuidedGenerationParams"] = params
            elif task_type == "IMAGE_VARIATION":
                body["imageVariationParams"] = params
            elif task_type == "INPAINTING":
                body["inPaintingParams"] = params
            elif task_type == "OUTPAINTING":
                body["outPaintingParams"] = params
            elif task_type == "BACKGROUND_REMOVAL":
                body["backgroundRemovalParams"] = params
            else:
                raise ValueError(f"Unsupported task type: {task_type}")

            # Generate images
            image_bytes_list = self._invoke_model(body)
            return self._save_images(image_bytes_list, task_type)

        except Exception as e:
            logger.error(f"Failed to generate {task_type}: {str(e)}", exc_info=True)
            raise

    # Legacy methods remain unchanged...
