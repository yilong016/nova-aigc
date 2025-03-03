import time
import boto3
import json
import base64
from PIL import Image
import io
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ImageToVideoPrompt:
    """Class for generating video prompts from images using AWS Bedrock."""
    
    def __init__(self, region_name=None, model_name=None):
        """Initialize the ImageToVideoPrompt generator.
        
        Args:
            region_name (str, optional): AWS region name. Defaults to 'us-east-1'.
            model_name (str, optional): Bedrock model name. Defaults to nova pro.
        """
        logger.info("Initializing ImageToVideoPrompt generator...")
        
        self.region_name = region_name or os.getenv('AWS_REGION', 'us-east-1')
        self.model_name = model_name or os.getenv('IMAGE_TO_PROMPTS_MODEL')
        
        self.bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=self.region_name
        )
        
        logger.info(f"Using region: {self.region_name}")
        logger.info(f"Using model: {self.model_name}")
        logger.info("ImageToVideoPrompt generator initialized successfully")

    def process_images(self, image_paths):
        """Process multiple images and return a list of image objects ready for the API.
        
        Args:
            image_paths (list): List of paths to image files
            
        Returns:
            list: List of image objects formatted for the Bedrock API
        """
        logger.debug(f"Processing {len(image_paths)} images")
        image_objects = []
        
        for image_path in image_paths:
            try:
                imagedata, image_type = self._image_base64_encoder(image_path)
                image_object = {
                    "image": {
                        "format": image_type,
                        "source": {
                            "bytes": imagedata
                        }
                    }
                }
                image_objects.append(image_object)
                logger.debug(f"Successfully processed image: {image_path}")
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {str(e)}")
                raise
                
        return image_objects

    def _image_base64_encoder(self, image_path, max_size=1568):
        """Encode an image to base64 and resize if necessary.
        
        Args:
            image_path (str): Path to the image file
            max_size (int, optional): Maximum dimension for resizing. Defaults to 1568.
            
        Returns:
            tuple: (image_bytes, image_format)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
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
        resized_bytes = resized_bytes.getvalue()

        return resized_bytes, img_format

    def generate_prompts(self, image_paths):
        """Generate video prompts for multiple images.
        
        Args:
            image_paths (list): List of paths to image files
            
        Returns:
            dict: Dictionary mapping image paths to generated prompts
        """
        if not image_paths:
            raise ValueError("No image paths provided")
            
        if len(image_paths) > 10:
            raise ValueError("Maximum of 10 images allowed")
            
        logger.info(f"Generating video prompts for {len(image_paths)} images")
        
        try:
            # Process images for API
            image_objects = self.process_images(image_paths)
            
            # System prompt
            system_text = '''
            你是一个专业图片分析助手，专注于提取图片的信息给用户。
            '''
            system_prompts = [{"text": system_text}]
            
            # User prompt
            text='''
#任务描述
你是一位专业旅游视觉导演，负责为Amazon Canvas Reel创建简洁有力的视频生成提示词。我会上传旅游行业相关的照片(如海岛、沙滩、美食、酒店、游乐场、自然风光等)，请为每张照片创建能使静态图像转化为动态视频的专业提示词。

#输出要求
- 每个提示词应简洁精炼，采用摄影术语描述平稳缓慢的运镜动作
- 清晰描述主体、环境和氛围元素
- 包含必要的技术参数(如4k, cinematic, photorealistic)
- 突出旅游体验的独特卖点
- 确保动作微妙、自然，避免剧烈变化

#输出格式
对于每张照片，请提供以下格式，不要添加任何其他格式修饰：
V[编号]:"[图片名称]; [平稳运镜动作] of [主体/场景描述]; [环境/氛围元素]; [技术参数]; [附加细节]"

#参考示例
V1: "white beach; Gentle aerial drift over pristine white sand beach with turquoise waters; palm trees swaying subtly; luxury resort visible in background; golden sunset light; 4k; cinematic; shallow depth of field"
V2: "night bar; Subtle push in on exotic cocktail with tropical garnish; beachfront bar setting; soft ocean waves in background; condensation droplets visible; warm evening light; 4k; photorealistic"
V3: "theme park; Slow steady tracking shot observing a family walking through colorful theme park; excited children pointing at attractions; confetti falling gently; sunny day; vibrant colors; 4k; cinematic; natural lighting"
V4: "luxury hotel; Smooth gradual pan across luxury hotel lobby; crystal chandeliers; marble floors with reflections; well-dressed guests checking in; soft ambient lighting; 4k; photorealistic; shallow focus"
V5: "view; Gentle first-person perspective slowly approaching a private balcony; panoramic mountain vista gradually revealed; steam rising from coffee cup in foreground; morning mist in valley below; birds flying; 4k; cinematic"
V6: "312; Minimal motion closeup of chef's hands preparing sushi; fresh ingredients; light steam rising; precision knife work; restaurant ambient lighting; 4k; photorealistic; macro details"
V7: "231; Slow deliberate orbit around ancient temple ruins; golden hour lighting; tourists exploring in distance; birds flying overhead; historical architecture details; 4k; cinematic; atmospheric"
V8: "12; Subtle push in on infinity pool merging with ocean horizon; gentle rippling water reflections; empty lounger with folded towel; cocktail on side table; sunset colors; 4k; cinematic"

#额外指南
- 使用平稳缓慢的运镜指令(gentle drift, subtle push, slow tracking, smooth pan, minimal motion, gradual等)
- 避免快速或大幅度的相机运动，保持镜头移动自然和微妙
- 添加视觉细节以增强真实感(如光线条件、天气状态、材质细节)
- 根据照片内容选择最能展现空间感和氛围的平稳运镜动作
- 控制每个提示词在40-60个单词左右，确保AI能有效处理
'''
            text_1 = '''
            #任务描述
            你是一位专业旅游视觉导演，负责为Amazon Canvas Reel创建简洁有力的视频生成提示词。我会上传旅游行业相关的照片(如海岛、沙滩、美食、酒店、游乐场、自然风光等)，请为每张照片创建能使静态图像转化为动态视频的专业提示词。

            #输出要求
            - 每个提示词应简洁精炼，采用摄影术语描述运镜动作
            - 清晰描述主体、环境和氛围元素
            - 包含必要的技术参数(如4k, cinematic, photorealistic)
            - 突出旅游体验的独特卖点

            #输出格式
            对于每张照片，请提供以下格式，不要添加任何其他格式修饰：
            V[编号]:"[图片名称]; [运镜动作] of [主体/场景描述]; [环境/氛围元素]; [技术参数]; [附加细节]"

            #参考示例
            V1: "white beach; Aerial dolly shot of pristine white sand beach with turquoise waters; palm trees swaying gently; luxury resort visible in background; golden sunset light; 4k; cinematic; shallow depth of field"
            V2: "night bar; Slow push in on exotic cocktail with tropical garnish; beachfront bar setting; soft ocean waves in background; condensation droplets visible; warm evening light; 4k; photorealistic"
            V3: "theme park; Tracking shot following a family walking through colorful theme park; excited children pointing at attractions; confetti falling; sunny day; vibrant colors; 4k; cinematic; natural lighting"
            V4: "luxury hotel; Cinematic pan across luxury hotel lobby; crystal chandeliers; marble floors with reflections; well-dressed guests checking in; soft ambient lighting; 4k; photorealistic; shallow focus"
            V5: "view; First person perspective walking onto a private balcony; panoramic mountain vista revealed; steam rising from coffee cup in foreground; morning mist in valley below; birds flying; 4k; cinematic"
            V6: "312; Slow motion closeup of chef's hands preparing sushi; fresh ingredients; steam rising; precision knife work; restaurant ambient lighting; 4k; photorealistic; macro details"
            V7: "231; Orbit shot around ancient temple ruins; golden hour lighting; tourists exploring in distance; birds flying overhead; historical architecture details; 4k; cinematic; atmospheric"
            V8: "12; Gentle push in on infinity pool merging with ocean horizon; rippling water reflections; empty lounger with folded towel; cocktail on side table; sunset colors; 4k; cinematic"

            #额外指南
            - 确保运镜指令清晰明确(dolly, pan, tracking, zoom, orbit, aerial等)
            - 添加视觉细节以增强真实感(如光线条件、天气状态、材质细节)
            - 根据照片内容选择最能展现空间感和氛围的运镜动作
            - 控制每个提示词在40-60个单词左右，确保AI能有效处理
            '''

            # Construct the messages with text and all images
            content = [{"text": text}]
            content.extend(image_objects)
            
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
            
            # Base inference parameters
            inference_config = {"temperature": 1}
            
            # Additional inference parameters
            #additional_model_fields = {"top_k": 200}
            
            logger.info("Calling Bedrock API for prompt generation...")
            response = self.bedrock_client.converse(
                modelId=self.model_name,
                messages=messages,
                inferenceConfig=inference_config,
                #additionalModelRequestFields=additional_model_fields,
            )
            
            # Extract the generated prompts
            result_text = response['output']['message']['content'][0]['text']
            logger.info("Successfully generated video prompts")
            logger.info(f"Generated prompts: {result_text}")
            # Parse the result text to extract prompts for each image
            return self._parse_prompts(result_text, image_paths)
            
        except Exception as e:
            logger.error(f"Error generating prompts: {str(e)}", exc_info=True)
            raise

    def _parse_prompts(self, result_text, image_paths):
        """Parse the result text to extract prompts for each image.
        
        Args:
            result_text (str): The text response from the API
            image_paths (list): List of image paths
            
        Returns:
            dict: Dictionary mapping image paths to generated prompts
        """
        # Split the text by lines
        lines = result_text.strip().split('\n')
        
        # Extract lines that start with V followed by a number
        prompt_lines = [line for line in lines if line.strip() and line.strip()[0] == 'V' and ':' in line]
        
        # Create a dictionary mapping image paths to prompts
        prompts = {}
        
        # Match prompts to images based on order
        for i, image_path in enumerate(image_paths):
            if i < len(prompt_lines):
                # Extract the prompt part (after the colon)
                prompt_parts = prompt_lines[i].split(':', 1)
                if len(prompt_parts) > 1:
                    prompt = prompt_parts[1].strip().strip('"')
                    prompts[image_path] = prompt
                else:
                    prompts[image_path] = ""
            else:
                # If we have more images than prompts, use empty string
                prompts[image_path] = ""
                
        return prompts

    def generate_prompt(self, image_path):
        """Generate a video prompt for a single image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Generated prompt for the image
        """
        prompts = self.generate_prompts([image_path])
        return prompts.get(image_path, "")


# Example usage
if __name__ == "__main__":
    try:
        # Initialize the generator
        generator = ImageToVideoPrompt()
        
        # Example image paths
        image_paths = [
            '/path/to/image1.jpg',
            '/path/to/image2.jpg',
            '/path/to/image3.jpg'
        ]
        
        # Generate prompts for multiple images
        prompts = generator.generate_prompts(image_paths)
        
        # Print the results
        for image_path, prompt in prompts.items():
            print(f"Image: {image_path}")
            print(f"Prompt: {prompt}")
            print()
            
    except Exception as e:
        print(f"Error: {str(e)}")
