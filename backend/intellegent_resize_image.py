import boto3
import io
import os
import logging
from PIL import Image
from typing import Dict, Optional, Tuple
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

class IntelligentImageResizer:
    """Intelligent image resizer that uses Bedrock Nova to identify main subjects and crop appropriately."""
    
    def __init__(self):
        """Initialize the intelligent image resizer with Bedrock client."""
        self.bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=os.getenv('AWS_REGION', 'us-west-2')
        )
        
        # Base parameters
        self.inference_config = {"temperature": 0.1}
        self.model_id = os.getenv('NOVA_MODEL_ID', 'us.amazon.nova-lite-v1:0')
        
    def _encode_image(self, image_path: str, max_size: int = 1568) -> Tuple[bytes, str]:
        """Process image for Nova API.
        
        Args:
            image_path (str): Path to the image file
            max_size (int): Maximum dimension for resizing
            
        Returns:
            tuple: (image_bytes, image_format)
        """
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
        img_format = img.format.lower()

        if width > max_size or height > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))

            img = img.resize((new_width, new_height), Image.LANCZOS)

        resized_bytes = io.BytesIO()
        img.save(resized_bytes, format=img_format)
        return resized_bytes.getvalue(), img_format
            
    def _analyze_image_content(self, image_path: str) -> Optional[Dict]:
        """Send image to Nova model for content analysis.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Dict: Model response containing subject location information
        """
        try:
            # Process image
            image_bytes, image_format = self._encode_image(image_path)
            
            # System prompt for image analysis
            system_prompts = [{
                "text": """You are an expert at analyzing images and identifying the main subject's location.
                Your task is to determine the optimal crop coordinates that will preserve the main subject
                while maintaining a 16:9 aspect ratio (1280x720). Focus on key elements like buildings,
                people, or important focal points."""
            }]
            
            # User message content
            content = [
                {
                    "text": """Analyze this image and identify the main subject's location.
                    Provide coordinates for a bounding box that will best capture the subject
                    while maintaining a 16:9 aspect ratio (1280x720)."""
                },
                {
                    "image": {
                        "format": image_format,
                        "source": {
                            "bytes": image_bytes
                        }
                    }
                }
            ]
            
            # Tool configuration for structured output
            tool_config = {
                "tools": [{
                    "toolSpec": {
                        "name": "get_crop_coordinates",
                        "description": "Return the coordinates for cropping the image",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    
                                            "left": {"type": "number"},
                                            "top": {"type": "number"},
                                            "width": {"type": "number"},
                                            "height": {"type": "number"}
                                },
                                "required": ["left", "top", "width", "height"]
                            }
                        }
                    }
                }]
            }
            
            # Call Nova model
            response = self.bedrock_client.converse(
                modelId=self.model_id,
                system=system_prompts,
                messages=[{'role': 'user', 'content': content}],
                inferenceConfig=self.inference_config,
                toolConfig=tool_config
            )
            
            # Extract crop coordinates from response
            content = response["output"]["message"]["content"]
            for item in content:
                if isinstance(item, dict) and "toolUse" in item:
                    print("tool use is: ",item["toolUse"]["input"])
                    return item["toolUse"]["input"]
            
            logger.warning("No crop coordinates found in model response")
            return None
            
        except Exception as e:
            logger.error(f"Error calling Nova model: {str(e)}")
            return None
            
    def _initial_proportional_resize(self, img: Image.Image, target_width: int = 1280, target_height: int = 720) -> Image.Image:
        """Perform initial proportional resize to get as close as possible to target dimensions.
        
        Args:
            img: Original PIL Image
            target_width: Target width (default: 1280)
            target_height: Target height (default: 720)
            
        Returns:
            PIL Image: Proportionally resized image
        """
        original_width, original_height = img.size
        target_ratio = target_width / target_height
        original_ratio = original_width / original_height
        
        logger.info(f"Initial dimensions: {original_width}x{original_height}, ratio: {original_ratio:.3f}")
        logger.info(f"Target dimensions: {target_width}x{target_height}, ratio: {target_ratio:.3f}")
        
        # Calculate resize ratios for both dimensions
        width_ratio = target_width / original_width
        height_ratio = target_height / original_height
        
        logger.info(f"Resize ratios - width: {width_ratio:.3f}, height: {height_ratio:.3f}")
        
        # Choose the ratio that will result in the image being at least as large as the target
        # in both dimensions while maintaining aspect ratio
        if original_ratio > target_ratio:
            # Image is wider than target ratio
            scale_ratio = height_ratio  # Scale by height to ensure no vertical whitespace
        else:
            # Image is taller than target ratio
            scale_ratio = width_ratio  # Scale by width to ensure no horizontal whitespace
            
        # Calculate new dimensions
        new_width = int(original_width * scale_ratio)
        new_height = int(original_height * scale_ratio)
        
        logger.info(f"Proportional resize dimensions: {new_width}x{new_height}, ratio: {new_width/new_height:.3f}")
        
        # Perform the resize
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def resize_image(self, image_path: str, output_path: str, target_width: int = 1280, target_height: int = 720) -> str:
        """Intelligently resize image by identifying main subject and cropping appropriately.
        
        Args:
            image_path (str): Path to input image
            output_path (str): Path to save resized image
            target_width (int): Target width (default: 1280)
            target_height (int): Target height (default: 720)
            
        Returns:
            str: Path to resized image
        """
        try:
            with Image.open(image_path) as img:
                original_width, original_height = img.size
                
                logger.info(f"\n=== Starting image resize process ===")
                logger.info(f"Input image: {image_path}")
                logger.info(f"Original dimensions: {original_width}x{original_height}, ratio: {original_width/original_height:.3f}")
                
                # If image already matches target dimensions, return original
                if original_width == target_width and original_height == target_height:
                    logger.info("✓ Image already matches target dimensions")
                    return image_path
                
                # Step 1: Initial proportional resize
                resized_img = self._initial_proportional_resize(img, target_width, target_height)
                resized_width, resized_height = resized_img.size
                
                # If the resized image matches target dimensions exactly, we're done
                if resized_width == target_width and resized_height == target_height:
                    resized_img.save(output_path, quality=95)
                    logger.info(f"✓ Perfect proportional resize achieved")
                    logger.info(f"Saved image to: {output_path}")
                    return output_path
                
                # Step 2: Get content analysis from Nova Lite for intelligent cropping
                # Save temporary resized image for analysis
                # Use same format as original image for temp file
                original_format = img.format if img.format else 'JPEG'
                temp_path = f"{os.path.splitext(output_path)[0]}_temp.{original_format.lower()}"
                resized_img.save(temp_path, format=original_format)
                analysis = self._analyze_image_content(temp_path)
                try:
                    os.remove(temp_path)  # Clean up temporary file
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file {temp_path}: {str(e)}")
                
                target_ratio = target_width / target_height
                
                if analysis:
                    # Use model-suggested crop box
                    crop_box = analysis
                    logger.info(f"\nNova analysis successful")
                    logger.info(f"Initial crop box - left: {crop_box['left']:.1f}, top: {crop_box['top']:.1f}, " 
                              f"width: {crop_box['width']:.1f}, height: {crop_box['height']:.1f}")
                    
                    # Ensure crop box maintains target aspect ratio
                    crop_width = crop_box['width']
                    crop_height = crop_box['height']
                    original_crop_ratio = crop_width / crop_height
                    if original_crop_ratio != target_ratio:
                        # Adjust width to match target ratio
                        crop_width = crop_height * target_ratio
                        crop_box['width'] = crop_width
                        logger.info(f"Adjusted crop ratio: {original_crop_ratio:.3f} -> {target_ratio:.3f}")
                    
                    # Ensure crop box is within image bounds
                    crop_box['left'] = max(0, min(crop_box['left'], resized_width - crop_width))
                    crop_box['top'] = max(0, min(crop_box['top'], resized_height - crop_height))
                    logger.info(f"Final crop box - left: {crop_box['left']:.1f}, top: {crop_box['top']:.1f}, "
                              f"width: {crop_box['width']:.1f}, height: {crop_box['height']:.1f}")
                    
                    # Crop based on Nova's analysis
                    cropped = resized_img.crop((
                        crop_box['left'],
                        crop_box['top'],
                        crop_box['left'] + crop_box['width'],
                        crop_box['top'] + crop_box['height']
                    ))
                else:
                    # Fallback to center crop if analysis fails
                    logger.info(f"\nNova analysis failed, using center crop")
                    
                    if resized_width > target_width:
                        # Crop excess width from center
                        left = (resized_width - target_width) // 2
                        cropped = resized_img.crop((left, 0, left + target_width, resized_height))
                    else:
                        # Crop excess height from center
                        top = (resized_height - target_height) // 2
                        cropped = resized_img.crop((0, top, target_width, top + target_height))
                
                # Ensure final resize to target dimensions
                final_image = cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)
                final_image.save(output_path, quality=95)
                logger.info(f"\n✓ Final output: {target_width}x{target_height}, ratio: {target_width/target_height:.3f}")
                logger.info(f"Saved to: {output_path}")
                logger.info(f"=== Resize process complete ===\n")
                return output_path
                
        except Exception as e:
            logger.error(f"Error resizing image: {str(e)}")
            raise
            
# Example usage
if __name__ == "__main__":
    try:
        resizer = IntelligentImageResizer()
        input_path = "./data/sample_image.jpg"  # Replace with your image path
        output_path = "./data/sample_image_resized.jpg"  # Replace with desired output path
        resized_path = resizer.resize_image(input_path, output_path)
        print(f"Successfully resized image: {resized_path}")
    except Exception as e:
        print(f"Error: {str(e)}")
