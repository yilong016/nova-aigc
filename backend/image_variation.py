import json
import boto3
import base64
import os
import logging
import imghdr
from typing import Dict
from PIL import Image
from io import BytesIO
from botocore.config import Config
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class NovaImageVariation:
    """Nova Image Variation Generator for creating variations of source images."""
    
    def __init__(self):
        """Initialize the Nova Image Variation Generator."""
        logger.info("Initializing NovaImageVariation...")
        
        self.bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name='us-east-1',
            config=Config(read_timeout=300)
        )
        self.output_dir = "./output/generated_image"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.model_id = "amazon.nova-canvas-v1:0"
            
        logger.info("NovaImageVariation initialized successfully")

    def _invoke_model(self, body: Dict) -> bytes:
        """Invoke the Bedrock model with the given body.

        Args:
            body (Dict): The request body

        Returns:
            bytes: Generated image bytes

        Raises:
            Exception: If image generation fails
        """
        try:
            logger.info("Calling Bedrock API for image variation...")
            response = self.bedrock_runtime.invoke_model(
                body=json.dumps(body),
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json"
            )
            
            response_body = json.loads(response.get("body").read())
            
            # Check for errors
            if "error" in response_body and response_body["error"]:
                error_msg = response_body["error"]
                logger.error(f"Image variation failed: {error_msg}")
                raise Exception(f"Image variation failed: {error_msg}")
            
            # Get image data
            base64_image = response_body.get("images")[0]
            return base64.b64decode(base64_image.encode('ascii'))
            
        except Exception as e:
            logger.error(f"Model invocation failed: {str(e)}", exc_info=True)
            raise

    def _save_image(self, image_bytes: bytes, prefix: str = "variation") -> str:
        """Save image bytes to file with timestamp.

        Args:
            image_bytes (bytes): Image data to save
            prefix (str, optional): Filename prefix. Defaults to "variation".

        Returns:
            str: Path to saved image
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(self.output_dir, f"{prefix}_image_{timestamp}.png")
        
        image = Image.open(BytesIO(image_bytes))
        image.save(output_path, format='PNG')
        
        logger.info(f"Image saved to: {output_path}")
        return output_path

    def _verify_image_format(self, image_path: str) -> None:
        """Verify that the image is in JPEG or PNG format.

        Args:
            image_path (str): Path to the image file

        Raises:
            ValueError: If the image is not in JPEG or PNG format
        """
        # Check if file exists
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")

        # Detect actual image type
        actual_type = imghdr.what(image_path)
        logger.info(f"Detected image type: {actual_type}")

        # Verify format
        if actual_type not in ['jpeg', 'png']:
            raise ValueError(f"Invalid image format. Expected JPEG or PNG, but got: {actual_type}")

    def generate_variation(
        self,
        image_path: str,
        prompt: str,
        negative_prompt: str = "bad quality, low resolution",
        similarity_strength: float = 0.7,
        width: int = 1024,
        height: int = 1024,
        num_images: int = 1,
        cfg_scale: float = 8.0
    ) -> str:
        """Generate image variation from source image.

        Args:
            image_path (str): Path to source image
            prompt (str): Text description for variation
            negative_prompt (str, optional): Negative prompt. Defaults to "bad quality, low resolution".
            similarity_strength (float, optional): Strength of similarity to source. Range 0.2-1.0. Defaults to 0.7.
            width (int, optional): Image width. Defaults to 1024.
            height (int, optional): Image height. Defaults to 1024.
            num_images (int, optional): Number of images. Defaults to 1.
            cfg_scale (float, optional): CFG scale parameter. Defaults to 8.0.

        Returns:
            str: Path to the generated variation image
        """
        try:
            logger.info("Starting image variation generation")
            logger.info(f"Using prompt: {prompt}")
            
            # Verify image format before proceeding
            self._verify_image_format(image_path)
            
            # Validate similarity strength
            if not 0.2 <= similarity_strength <= 1.0:
                raise ValueError("similarity_strength must be between 0.2 and 1.0")
            
            # Read and encode source image
            with open(image_path, "rb") as f:
                input_image = base64.b64encode(f.read()).decode('utf8')
            
            body = {
                "taskType": "IMAGE_VARIATION",
                "imageVariationParams": {
                    "text": prompt,
                    "negativeText": negative_prompt,
                    "images": [input_image],
                    "similarityStrength": similarity_strength
                },
                "imageGenerationConfig": {
                    "numberOfImages": num_images,
                    "height": height,
                    "width": width,
                    "cfgScale": cfg_scale
                }
            }

            image_bytes = self._invoke_model(body)
            return self._save_image(image_bytes)
            
        except Exception as e:
            logger.error(f"Failed to generate image variation: {str(e)}", exc_info=True)
            raise


# Example usage
if __name__ == "__main__":
    try:
        # Initialize generator
        generator = NovaImageVariation()
        
        # Generate image variation
        variation_path = generator.generate_variation(
            image_path="./data/comic/origin.jpeg",
            prompt="A minimalist dog portrait featuring a cream-colored face with bold black outlines, cartoon style.",
            negative_prompt="bad quality, low resolution, cartoon",
            similarity_strength=1,
            width=512,
            height=512
        )
        logger.info(f"Image variation generated: {variation_path}")
        
        # Show generated image
        Image.open(variation_path).show()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
