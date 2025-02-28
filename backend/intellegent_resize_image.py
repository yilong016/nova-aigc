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
            
    def _initial_proportional_resize(self, img: Image.Image, output_base_path: str, target_width: int = 1280, target_height: int = 720) -> Image.Image:
        """Perform initial proportional resize to get as close as possible to target dimensions.
        
        Args:
            img: Original PIL Image
            output_base_path: Base path for saving intermediate images
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
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Save the proportionally resized image
        prop_resize_path = f"{output_base_path}_1_proportional_resize.jpg"
        resized_img.save(prop_resize_path, quality=95)
        logger.info(f"Saved proportionally resized image to: {prop_resize_path}")
        
        return resized_img

    def resize_image(self, image_path: str, output_path: str, target_width: int = 1280, target_height: int = 720, save_steps: bool = True) -> str:
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
                
                # Get base path for intermediate images
                output_base_path = os.path.splitext(output_path)[0]
                
                # Save original image copy
                original_copy_path = f"{output_base_path}_0_original.jpg"
                img.save(original_copy_path, quality=95)
                logger.info(f"Saved original image copy to: {original_copy_path}")
                
                # If image already matches target dimensions, return original
                if original_width == target_width and original_height == target_height:
                    logger.info("✓ Image already matches target dimensions")
                    return image_path
                
                # Step 1: Initial proportional resize
                resized_img = self._initial_proportional_resize(img, output_base_path, target_width, target_height)
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
                temp_path = f"{os.path.splitext(output_path)[0]}_2_for_analysis.{original_format.lower()}"
                resized_img.save(temp_path, format=original_format)
                logger.info(f"Saved image for analysis to: {temp_path}")
                analysis = self._analyze_image_content(temp_path)
                # Keep the temporary file for inspection
                
                target_ratio = target_width / target_height
                
                if analysis:
                    # Use model-suggested crop box
                    crop_box = analysis
                    logger.info(f"\nNova analysis successful")
                    logger.info(f"Initial crop box - left: {crop_box['left']:.1f}, top: {crop_box['top']:.1f}, " 
                              f"width: {crop_box['width']:.1f}, height: {crop_box['height']:.1f}")
                    
                    # Calculate center point of the subject box identified by Nova
                    subject_width = crop_box['width']
                    subject_height = crop_box['height']
                    subject_center_x = crop_box['left'] + subject_width / 2
                    subject_center_y = crop_box['top'] + subject_height / 2
                    
                    logger.info(f"Subject center point: x={subject_center_x:.1f}, y={subject_center_y:.1f}")
                    
                    # Create a new crop box with target dimensions (1280x720 aspect ratio)
                    # Calculate the dimensions of the new crop box
                    if resized_width / resized_height > target_ratio:
                        # Image is wider than target ratio
                        crop_height = min(resized_height, target_height)
                        crop_width = crop_height * target_ratio
                    else:
                        # Image is taller than target ratio
                        crop_width = min(resized_width, target_width)
                        crop_height = crop_width / target_ratio
                    
                    logger.info(f"New crop dimensions: {crop_width:.1f}x{crop_height:.1f}, ratio: {crop_width/crop_height:.3f}")
                    
                    # Calculate how much we need to expand in each direction from the subject center
                    half_width = crop_width / 2
                    half_height = crop_height / 2
                    
                    # Calculate initial crop box boundaries based on subject center
                    left_boundary = subject_center_x - half_width
                    right_boundary = subject_center_x + half_width
                    top_boundary = subject_center_y - half_height
                    bottom_boundary = subject_center_y + half_height
                    
                    # Check if any boundary exceeds image dimensions and adjust while maintaining aspect ratio
                    if left_boundary < 0:
                        # Shift the entire crop box right
                        shift_right = -left_boundary
                        left_boundary = 0
                        right_boundary += shift_right
                        # If right boundary now exceeds image width, we need to shrink the crop
                        if right_boundary > resized_width:
                            # Shrink width and recalculate height to maintain aspect ratio
                            crop_width = resized_width
                            crop_height = crop_width / target_ratio
                            # Recenter vertically
                            half_height = crop_height / 2
                            top_boundary = subject_center_y - half_height
                            bottom_boundary = subject_center_y + half_height
                    
                    elif right_boundary > resized_width:
                        # Shift the entire crop box left
                        shift_left = right_boundary - resized_width
                        right_boundary = resized_width
                        left_boundary -= shift_left
                        # If left boundary now becomes negative, we need to shrink the crop
                        if left_boundary < 0:
                            # Shrink width and recalculate height to maintain aspect ratio
                            crop_width = resized_width
                            crop_height = crop_width / target_ratio
                            # Recenter vertically
                            half_height = crop_height / 2
                            top_boundary = subject_center_y - half_height
                            bottom_boundary = subject_center_y + half_height
                    
                    if top_boundary < 0:
                        # Shift the entire crop box down
                        shift_down = -top_boundary
                        top_boundary = 0
                        bottom_boundary += shift_down
                        # If bottom boundary now exceeds image height, we need to shrink the crop
                        if bottom_boundary > resized_height:
                            # Shrink height and recalculate width to maintain aspect ratio
                            crop_height = resized_height
                            crop_width = crop_height * target_ratio
                            # Recenter horizontally
                            half_width = crop_width / 2
                            left_boundary = subject_center_x - half_width
                            right_boundary = subject_center_x + half_width
                    
                    elif bottom_boundary > resized_height:
                        # Shift the entire crop box up
                        shift_up = bottom_boundary - resized_height
                        bottom_boundary = resized_height
                        top_boundary -= shift_up
                        # If top boundary now becomes negative, we need to shrink the crop
                        if top_boundary < 0:
                            # Shrink height and recalculate width to maintain aspect ratio
                            crop_height = resized_height
                            crop_width = crop_height * target_ratio
                            # Recenter horizontally
                            half_width = crop_width / 2
                            left_boundary = subject_center_x - half_width
                            right_boundary = subject_center_x + half_width
                    
                    # Final check to ensure boundaries are within image dimensions
                    left_boundary = max(0, left_boundary)
                    top_boundary = max(0, top_boundary)
                    right_boundary = min(resized_width, right_boundary)
                    bottom_boundary = min(resized_height, bottom_boundary)
                    
                    # Update crop dimensions based on final boundaries
                    crop_width = right_boundary - left_boundary
                    crop_height = bottom_boundary - top_boundary
                    
                    logger.info(f"Adjusted crop dimensions: {crop_width:.1f}x{crop_height:.1f}, ratio: {crop_width/crop_height:.3f}")
                    
                    # Set the final crop box coordinates
                    crop_left = left_boundary
                    crop_top = top_boundary
                    
                    # Update crop box with new coordinates
                    crop_box = {
                        'left': crop_left,
                        'top': crop_top,
                        'width': crop_width,
                        'height': crop_height
                    }
                    
                    logger.info(f"Final crop box - left: {crop_box['left']:.1f}, top: {crop_box['top']:.1f}, "
                              f"width: {crop_box['width']:.1f}, height: {crop_box['height']:.1f}")
                    
                    # Save a visualization of the crop box on the resized image
                    crop_viz_img = resized_img.copy()
                    try:
                        from PIL import ImageDraw
                        draw = ImageDraw.Draw(crop_viz_img)
                        draw.rectangle(
                            [
                                crop_box['left'], 
                                crop_box['top'], 
                                crop_box['left'] + crop_box['width'], 
                                crop_box['top'] + crop_box['height']
                            ],
                            outline=(255, 0, 0),
                            width=3
                        )
                        crop_viz_path = f"{output_base_path}_2b_crop_visualization.jpg"
                        crop_viz_img.save(crop_viz_path, quality=95)
                        logger.info(f"Saved crop box visualization to: {crop_viz_path}")
                    except Exception as e:
                        logger.warning(f"Could not create crop visualization: {str(e)}")
                    
                    # Crop based on Nova's analysis
                    cropped = resized_img.crop((
                        crop_box['left'],
                        crop_box['top'],
                        crop_box['left'] + crop_box['width'],
                        crop_box['top'] + crop_box['height']
                    ))
                    
                    # Save the cropped image
                    cropped_path = f"{output_base_path}_3a_nova_cropped.jpg"
                    cropped.save(cropped_path, quality=95)
                    logger.info(f"Saved Nova-cropped image to: {cropped_path}")
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
                        cropped = resized_img.crop((0, top, resized_width, top + target_height))
                    
                    # Save the center-cropped image
                    cropped_path = f"{output_base_path}_3b_center_cropped.jpg"
                    cropped.save(cropped_path, quality=95)
                    logger.info(f"Saved center-cropped image to: {cropped_path}")
                
                # Ensure final resize to target dimensions
                final_image = cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)
                
                # Save pre-final image (before saving to output_path)
                prefinal_path = f"{output_base_path}_4_final_resize.jpg"
                final_image.save(prefinal_path, quality=95)
                logger.info(f"Saved final resized image to: {prefinal_path}")
                final_image.save(output_path, quality=95)
                logger.info(f"\n✓ Final output: {target_width}x{target_height}, ratio: {target_width/target_height:.3f}")
                logger.info(f"Saved to: {output_path}")
                logger.info(f"=== Resize process complete ===\n")
                return output_path
                
        except Exception as e:
            logger.error(f"Error resizing image: {str(e)}")
            raise
            
def test_resize(input_path, output_dir=None, target_width=1280, target_height=720):
    """Test function to resize an image and save intermediate steps.
    
    Args:
        input_path (str): Path to input image
        output_dir (str, optional): Directory to save output images. If None, uses same directory as input
        target_width (int, optional): Target width. Defaults to 1280.
        target_height (int, optional): Target height. Defaults to 720.
    """
    try:
        # Create output directory if it doesn't exist
        if output_dir is None:
            output_dir = os.path.dirname(input_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename without extension
        base_filename = os.path.basename(input_path)
        name_without_ext = os.path.splitext(base_filename)[0]
        
        # Create output path
        output_path = os.path.join(output_dir, f"{name_without_ext}_resized.jpg")
        
        # Initialize resizer and process image
        resizer = IntelligentImageResizer()
        resized_path = resizer.resize_image(input_path, output_path, target_width, target_height)
        
        print(f"\nImage processing complete!")
        print(f"Original image: {input_path}")
        print(f"Resized image: {resized_path}")
        print(f"Intermediate images saved with prefixes in: {output_dir}")
        
        return resized_path
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Use command line arguments if provided
        input_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        target_width = int(sys.argv[3]) if len(sys.argv) > 3 else 1280
        target_height = int(sys.argv[4]) if len(sys.argv) > 4 else 720
        
        test_resize(input_path, output_dir, target_width, target_height)
    else:
        # Default example
        try:
            input_path = "/Users/yilongl/Documents/03_Accounts/01-Klook/GenAI/video/冲绳自有图/3.jpg"  # Replace with your image path
            output_dir = "/Users/yilongl/Documents/03_Accounts/01-Klook/GenAI/video/冲绳自有图/"  # Replace with desired output directory
            os.makedirs(output_dir, exist_ok=True)
            
            test_resize(input_path, output_dir)
            print("\nUsage: python intellegent_resize_image.py <input_path> [output_dir] [target_width] [target_height]")
        except Exception as e:
            print(f"Error: {str(e)}")
