import boto3
import base64
from PIL import Image
import io
import os
from dotenv import load_dotenv
from nova_reel_prompts import (
    TEXT_TO_VIDEO_SYSTEM,
    TEXT_TO_VIDEO_PROMPT,
    IMAGE_TO_VIDEO_SYSTEM,
    IMAGE_TO_VIDEO_PROMPT
)
from nova_canvas_prompts import TEXT_TO_IMAGE_SYSTEM, TEXT_TO_IMAGE_PROMPT

# Load environment variables
load_dotenv()

class BasePromptOptimizer:
    """Base class for prompt optimization using AWS Bedrock Nova."""
    
    def __init__(self):
        """Initialize the Prompt Optimizer."""
        self.bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=os.getenv('AWS_REGION', 'us-west-2')
        )
        
        # Base parameters
        self.inference_config = {"temperature": 0.5}
        self.model_id = os.getenv('NOVA_MODEL_ID', 'us.amazon.nova-pro-v1:0')

    def _encode_image(self, image_path: str, max_size=1568):
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

    def _find_tool_use_response(self, response):
        """Find the toolUse response in the content array.
        
        Args:
            response (dict): Nova API response
            
        Returns:
            str: Optimized prompt from toolUse, or None if not found
        """
        content = response["output"]["message"]["content"]
        for item in content:
            if isinstance(item, dict) and "toolUse" in item:
                return item["toolUse"]["input"]["optimized_prompt"]
        return None

    def _optimize(self, text: str, system_text: str, prompt_template: str, image_path: str = None) -> str:
        """Core optimization logic using Nova.
        
        Args:
            text (str): Original text prompt
            system_text (str): System prompt to use
            prompt_template (str): Prompt template to use
            image_path (str, optional): Path to input image
            
        Returns:
            str: Optimized text prompt
        """
        # Format system prompts
        system_prompts = [{"text": system_text}]

        # Prepare message content
        content = [{"text": prompt_template.format(text=text)}]
        if image_path:
            image_bytes, image_format = self._encode_image(image_path)
            content.append({
                "image": {
                    "format": image_format,
                    "source": {
                        "bytes": image_bytes
                    }
                }
            })

        # Tool configuration for structured output
        tool_config = {
            "tools": [{
                "toolSpec": {
                    "name": "optimize_prompt",
                    "description": "Optimize the text prompt for generation",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "optimized_prompt": {"type": "string"}
                            },
                            "required": ["optimized_prompt"]
                        }
                    }
                }
            }]
        }

        # Call Nova
        response = self.bedrock_client.converse(
            modelId=self.model_id,
            system=system_prompts,
            messages=[{'role': 'user', 'content': content}],
            inferenceConfig=self.inference_config,
            toolConfig=tool_config
        )

        print(f'Nova response: {response}')
        
        # Extract optimized prompt from response
        optimized_prompt = self._find_tool_use_response(response)
        return optimized_prompt if optimized_prompt else text


class ReelPromptOptimizer(BasePromptOptimizer):
    """Handles prompt optimization for Nova Reel (video generation)."""

    def optimize_prompt(self, text: str, image_path: str = None) -> str:
        """Optimize prompt for video generation, optionally with image context.
        
        Args:
            text (str): Original text prompt
            image_path (str, optional): Path to input image. If provided, uses image context.
            
        Returns:
            str: Optimized text prompt
        """
        if image_path:
            return self._optimize(text, IMAGE_TO_VIDEO_SYSTEM, IMAGE_TO_VIDEO_PROMPT, image_path)
        return self._optimize(text, TEXT_TO_VIDEO_SYSTEM, TEXT_TO_VIDEO_PROMPT)


class CanvasPromptOptimizer(BasePromptOptimizer):
    """Handles prompt optimization for Nova Canvas (image generation)."""

    def optimize_prompt(self, text: str) -> str:
        """Optimize prompt for image generation.
        
        Args:
            text (str): Original text prompt
            
        Returns:
            str: Optimized text prompt
        """
        return self._optimize(text, TEXT_TO_IMAGE_SYSTEM, TEXT_TO_IMAGE_PROMPT)


# For backwards compatibility
PromptOptimizer = ReelPromptOptimizer  # Default to ReelPromptOptimizer for existing code
