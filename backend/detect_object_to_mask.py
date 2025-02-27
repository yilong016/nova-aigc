import base64
import boto3
import io
import json
import re
import os
from PIL import Image, ImageDraw
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get output paths from environment variables or use defaults
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'output')
MASK_DIR = os.getenv('MASK_DIR', os.path.join(OUTPUT_DIR, 'mask'))
UPLOAD_DIR = os.getenv('UPLOAD_DIR', os.path.join(OUTPUT_DIR, 'uploads'))

# Initialize Amazon Bedrock client
modelId = "us.amazon.nova-lite-v1:0"
accept = "application/json"
contentType = "application/json"

try:
    bedrock_rt = boto3.client("bedrock-runtime", region_name="us-east-1")
except Exception as e:
    print(f"Error initializing Bedrock client: {e}")
    print("Please ensure AWS credentials are configured")

def safe_json_load(json_string):
    try:
        json_string = re.sub(r"\s", "", json_string)
        json_string = re.sub(r"\(", "[", json_string)
        json_string = re.sub(r"\)", "]", json_string)
        bbox_set = {}
        for b in re.finditer(r"\[\d+,\d+,\d+,\d+\]", json_string):
            if b.group(0) in bbox_set:
                json_string = json_string[:bbox_set[b.group(0)][1]] + "}]"
                break
            bbox_set[b.group(0)] = (b.start(), b.end())
        else:
            if bbox_set:
                json_string = json_string[:bbox_set[b.group(0)][1]] + "}]"
        json_string = re.sub(r"\]\},\]$", "]}]", json_string)
        json_string = re.sub(r"\]\],\[\"", "]},{\"", json_string)
        json_string = re.sub(r"\]\],\[\{\"", "]},{\"", json_string)
        json_string = re.sub(r"\",\[", "\":[", json_string)
        return json.loads(json_string)
    except Exception as e:
        print(json_string)
        print(e)
        return []

def ensure_directories():
    """Ensure all required directories exist"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

def detect_and_mask(image_path, object_name, model_choice="us.amazon.nova-lite-v1:0", image_short_size=480):
    """
    Detect specified object in image and create a mask where the object is black and background is white.
    
    Args:
        image_path (str): Path to the input image
        object_name (str): Name of the object to detect
        model_choice (str): Model ID to use
        image_short_size (int): Size to resize the shorter edge of the image to
        
    Returns:
        tuple: (original image with detection, mask image, detection coordinates, mask_path)
    """
    try:
        # Ensure output directories exist
        ensure_directories()
        
        # Load and process image
        image_pil = Image.open(image_path)
        original_size = image_pil.size
        
        # Calculate resize ratio
        width, height = image_pil.size
        ratio = image_short_size / min(width, height)
        resize_width = round(ratio * width)
        resize_height = round(ratio * height)
        
        # Resize image for detection
        resized_image = image_pil.resize((resize_width, resize_height), resample=Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        resized_image.save(buffer, format="webp", quality=90)
        image_data = buffer.getvalue()
        
        # Prepare prompt
        prompts = f"""
Detect bounding box of objects in the image, only detect "{object_name.lower()}" category objects with high confidence, output in a list of bounding box format.

Output example:
[
    {{"{object_name.lower().replace(' ', '_')}": [x1, y1, x2, y2]}},
    ...
]
"""
        
        prefill = """
[
    {"
""".strip("\n")
        
        # Prepare request
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "image": {
                            "format": 'webp',
                            "source": {
                                "bytes": image_data,
                            }
                        }
                    },
                    {
                        "text": prompts
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "text": prefill
                    },
                ],
            }
        ]
        
        # Call model
        response = bedrock_rt.converse(
            modelId=model_choice,
            messages=messages,
            inferenceConfig={
                "temperature": 0.0,
                "maxTokens": 1024,
            },
        )
        
        output = prefill + response.get('output')["message"]["content"][0]["text"]
        result = safe_json_load(output)
        
        # Create detection visualization
        detection_image = resized_image.copy()
        draw = ImageDraw.Draw(detection_image)
        
        # Create mask image (white background)
        mask_image = Image.new('RGB', resized_image.size, 'white')
        mask_draw = ImageDraw.Draw(mask_image)
        
        detected_coords = []
        
        for item in result:
            label = next(iter(item)).strip()
            if label == "others" or label == "other":
                continue
                
            bbox = item[label]
            x1, y1, x2, y2 = bbox
            
            if x1 >= x2 or y1 >= y2:
                continue
            
            # Convert coordinates to image space
            w, h = resized_image.size
            x1 = x1 / 1000 * w
            x2 = x2 / 1000 * w
            y1 = y1 / 1000 * h
            y2 = y2 / 1000 * h
            
            bbox = (x1, y1, x2, y2)
            bbox = list(map(round, bbox))
            detected_coords.append(bbox)
            
            # Draw detection box
            draw.rectangle(bbox, outline='blue', width=2)
            
            # Draw black mask for detected object
            mask_draw.rectangle(bbox, fill='black')
        
        # Resize mask back to original size if needed
        if resized_image.size != original_size:
            detection_image = detection_image.resize(original_size, Image.Resampling.LANCZOS)
            mask_image = mask_image.resize(original_size, Image.Resampling.LANCZOS)
        
        # Generate unique filename based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mask_filename = os.path.join(MASK_DIR, f"{object_name.replace(' ', '_')}_{timestamp}.png")
        
        # Save mask image
        mask_image.save(mask_filename)
        
        return detection_image, mask_image, detected_coords, mask_filename
        
    except Exception as e:
        print(f"Error during detection and masking: {str(e)}")
        return None, None, [], None

if __name__ == "__main__":
    # Test the function with environment variables
    test_image = os.path.join(os.getenv('TEST_DIR', 'test_images'), 'test.png')
    if os.path.exists(test_image):
        detection_img, mask_img, coords, mask_path = detect_and_mask(test_image, "bottle")
        if detection_img and mask_img:
            print(f"Detection coordinates: {coords}")
            print(f"Mask saved to: {mask_path}")
