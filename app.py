import gradio as gr
import os
import logging
import csv
from backend.video_generator import NovaVideoGenerator
from backend.image_generator import NovaImageGenerator
from backend.image_variation import NovaImageVariation
from backend.prompt_optimizer import PromptOptimizer, CanvasPromptOptimizer, ImageToPrompt
from backend.intellegent_resize_image import IntelligentImageResizer
from backend.detect_object_to_mask import detect_and_mask, ensure_directories
from backend.image_to_video_prompt import ImageToVideoPrompt
from backend.program_maker import VideoMaker
import time
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional, Union
import json

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize the generators and analyzers
logger.info("Initializing generators, optimizers, and analyzers...")
video_generator = NovaVideoGenerator()
image_to_prompt = ImageToPrompt()
image_generator = NovaImageGenerator()
image_variation = NovaImageVariation()
video_prompt_optimizer = PromptOptimizer()
image_prompt_optimizer = CanvasPromptOptimizer()
image_to_video_prompt = ImageToVideoPrompt()
video_maker = VideoMaker()

def analyze_image_to_prompt(image: str) -> str:
    """Analyze an image to generate a detailed prompt for image generation"""
    try:
        logger.info(f"Starting image analysis...")
        logger.info(f"Input image path: {image}")
        
        generated_prompt = image_to_prompt.get_prompt_from_image(image)
        logger.info("Image analysis completed successfully")
        return generated_prompt
    except Exception as e:
        logger.error(f"Error during image analysis: {str(e)}", exc_info=True)
        return f"Error analyzing image: {str(e)}"

def optimize_video_prompt(text: str, image: Optional[str] = None) -> str:
    """Optimize the input prompt for video generation"""
    try:
        logger.info(f"Starting video prompt optimization...")
        logger.info(f"Input text: {text}")
        if image:
            logger.info(f"Input image path: {image}")
        
        optimized = video_prompt_optimizer.optimize_prompt(text, image)
        logger.info(f"Prompt optimization completed. Original: '{text}' -> Optimized: '{optimized}'")
        return optimized
    except Exception as e:
        logger.error(f"Error during prompt optimization: {str(e)}", exc_info=True)
        return f"Error optimizing prompt: {str(e)}"

def optimize_image_prompt(text: str) -> str:
    """Optimize the input prompt for image generation"""
    try:
        logger.info(f"Starting image prompt optimization...")
        logger.info(f"Input text: {text}")
        
        optimized = image_prompt_optimizer.optimize_prompt(text)
        logger.info(f"Prompt optimization completed. Original: '{text}' -> Optimized: '{optimized}'")
        return optimized
    except Exception as e:
        logger.error(f"Error during prompt optimization: {str(e)}", exc_info=True)
        return f"Error optimizing prompt: {str(e)}"

def process_image_for_video(image_path: str) -> str:
    """Process image to meet video requirements (1280x720)"""
    try:
        logger.info(f"Processing image for video: {image_path}")
        resizer = IntelligentImageResizer()
        
        # Create output path
        directory = os.path.dirname(image_path)
        filename = os.path.basename(image_path)
        base_name, ext = os.path.splitext(filename)
        output_path = os.path.join(directory, f"{base_name}_processed{ext}")
        
        # Resize image
        processed_path = resizer.resize_image(
            image_path=image_path,
            output_path=output_path,
            target_width=1280,
            target_height=720
        )
        
        logger.info(f"Image processed successfully: {processed_path}")
        return processed_path
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise

def generate_video(text: str, image: Optional[str] = None, progress: Optional[gr.Progress] = gr.Progress()) -> str:
    """Generate video from text or image+text"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        logger.info(f"Starting video generation at {timestamp}")
        logger.info(f"Using prompt: {text}")
        
        # Process image if provided
        processed_image = None
        if image:
            logger.info(f"Processing input image: {image}")
            processed_image = process_image_for_video(image)
            progress(0.2, desc="Image processing completed")
            logger.info(f"Using processed image: {processed_image}")

        progress(0.3, desc="Starting video generation...")
        response = video_generator.generate_video(text=text, input_image_path=processed_image)
        
        if "invocationArn" not in response:
            logger.error("Failed to get invocation ARN from response")
            return "Failed to start video generation"
            
        invocation_arn = response["invocationArn"]
        logger.info(f"Generation job started with ARN: {invocation_arn}")
        
        check_count = 0
        while True:
            check_count += 1
            logger.info(f"Checking job status (attempt {check_count})...")
            
            status = video_generator.get_job_status(invocation_arn)
            current_status = status["status"]
            
            logger.info(f"Current status: {current_status}")
            
            if current_status == "InProgress":
                progress_val = min(0.05 + (check_count * 0.03), 0.9)
                progress(progress_val, desc=f"Generation in progress... (Status: {current_status})")
            
            if current_status == "Failed":
                error_msg = status.get('failure_message', 'Unknown error')
                logger.error(f"Generation failed: {error_msg}")
                return f"Generation failed: {error_msg}"
                
            if current_status == "Completed":
                progress(0.95, desc="Downloading generated video...")
                
                video_uri = status["video_uri"]
                local_dir = os.getenv('VIDEO_OUTPUT_DIR', './output/generated_videos')
                logger.info(f"Generation completed. Downloading video from {video_uri}")
                
                try:
                    video_path = video_generator.download_video(video_uri, local_dir)
                    logger.info(f"Video successfully downloaded to: {video_path}")
                    progress(1.0, desc="Video generation completed!")
                    return video_path
                except Exception as e:
                    logger.error(f"Error downloading video: {str(e)}", exc_info=True)
                    return f"Error downloading video: {str(e)}"
                    
            time.sleep(10)
            
    except Exception as e:
        logger.error(f"Error during video generation: {str(e)}", exc_info=True)
        return f"Error generating video: {str(e)}"

def generate_image(text: str, progress: Optional[gr.Progress] = gr.Progress()) -> str:
    """Generate image from text prompt"""
    try:
        logger.info("Starting image generation")
        logger.info(f"Using prompt: {text}")
        
        progress(0.2, desc="Generating image...")
        
        # Get dimensions from environment variables or use defaults
        dimensions = os.getenv('VIDEO_DEFAULT_DIMENSION', '1280x720').split('x')
        width = int(dimensions[0])
        height = int(dimensions[1])
        
        image_path = image_generator.generate_image(
            prompt=text,
            width=width,
            height=height
        )
        
        progress(1.0, desc="Image generation completed!")
        logger.info(f"Image generated successfully: {image_path}")
        return image_path
        
    except Exception as e:
        logger.error(f"Error during image generation: {str(e)}", exc_info=True)
        return f"Error generating image: {str(e)}"

def handle_image_generation(
    task_type: str,
    text: str,
    negative_text: str,
    width: int,
    height: int,
    quality: str,
    cfg_scale: float,
    seed: int,
    num_images: int,
    conditioning_input: Optional[str] = None,
    variation_input: Optional[str] = None,
    inpainting_input: Optional[str] = None,
    outpainting_input: Optional[str] = None,
    background_removal_input: Optional[str] = None,
    control_mode: Optional[str] = None,
    control_strength: Optional[float] = None,
    colors: Optional[str] = None,
    color_guided_reference: Optional[str] = None,
    similarity_strength: Optional[float] = None,
    mask_prompt: Optional[str] = None,
    mask_image: Optional[str] = None,
    outpainting_mode: Optional[str] = None,
    progress: Optional[gr.Progress] = gr.Progress()
) -> str:
    """Handle all image generation tasks"""
    try:
        logger.info(f"Starting {task_type} generation")
        progress(0.2, desc=f"Processing {task_type}...")

        # Prepare task-specific configuration
        if task_type == "BACKGROUND_REMOVAL":
            # Background removal doesn't need any config
            image_generation_config = {}
            
        elif task_type in ["INPAINTING", "OUTPAINTING"]:
            # Inpainting/Outpainting only need these parameters
            image_generation_config = {
                "numberOfImages": num_images,
                "quality": quality,
                "cfgScale": cfg_scale,
                "seed": seed
            }
            
        elif task_type == "IMAGE_VARIATION":
            # Image variation doesn't use quality
            image_generation_config = {
                "numberOfImages": num_images,
                "height": height,
                "width": width,
                "cfgScale": cfg_scale,
                "seed": seed
            }
            
        else:  # TEXT_IMAGE and COLOR_GUIDED_GENERATION
            # These tasks use all configuration parameters
            image_generation_config = {
                "width": width,
                "height": height,
                "quality": quality,
                "cfgScale": cfg_scale,
                "seed": seed,
                "numberOfImages": num_images
            }

        # Handle different task types
        if task_type in ["TEXT_IMAGE", "TEXT_IMAGE with conditioning"]:
            params = {
                "text": text,
                "negativeText": negative_text
            }
            # Add conditioning parameters if it's TEXT_IMAGE with conditioning
            if task_type == "TEXT_IMAGE with conditioning" and conditioning_input and control_mode:
                params["conditionImage"] = conditioning_input
                params["controlMode"] = control_mode
                params["controlStrength"] = control_strength

        elif task_type == "COLOR_GUIDED_GENERATION":
            # Parse the colors string into a JSON array
            try:
                if colors and colors.strip():
                    # Handle both comma-separated format and JSON string format
                    if colors.strip().startswith('['):
                        color_array = json.loads(colors)
                    else:
                        color_array = [c.strip() for c in colors.split(',')]
                else:
                    color_array = []
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing colors: {str(e)}")
                raise ValueError(f"Invalid color format. Please provide colors as comma-separated hex values or a JSON array: {str(e)}")

            params = {
                "colors": color_array,  # Now properly formatted as a JSON array
                "text": text,
                "negativeText": negative_text
            }
            # Only add referenceImage if it has a value
            if color_guided_reference:
                params["referenceImage"] = color_guided_reference

        elif task_type == "IMAGE_VARIATION":
            params = {
                "images": [variation_input],
                "similarityStrength": similarity_strength,
                "text": text,
                "negativeText": negative_text
            }

        elif task_type == "INPAINTING":
            params = {
                "image": inpainting_input,
                "negativeText": negative_text
            }
            
            # Only add text if it's not empty (allowing for object removal without replacement)
            if text and text.strip():
                params["text"] = text
            
            # Validate and add mask parameter for inpainting
            if mask_prompt and mask_image:
                raise ValueError("Please provide either maskPrompt or maskImage for inpainting, but not both")
            elif not mask_prompt and not mask_image:
                raise ValueError("Either maskPrompt or maskImage must be provided for inpainting")
            elif mask_prompt:
                params["maskPrompt"] = mask_prompt
            else:
                params["maskImage"] = mask_image
                
        elif task_type == "OUTPAINTING":
            # For outpainting, text is required
            if not text or not text.strip():
                raise ValueError("Text prompt is required for outpainting")
                
            params = {
                "image": outpainting_input,
                "text": text,
                "negativeText": negative_text,
                "outPaintingMode": outpainting_mode
            }
            
            # Validate and add mask parameter for outpainting
            if mask_prompt and mask_image:
                raise ValueError("Please provide either maskPrompt or maskImage for outpainting, but not both")
            elif not mask_prompt and not mask_image:
                raise ValueError("Either maskPrompt or maskImage must be provided for outpainting")
            elif mask_prompt:
                params["maskPrompt"] = mask_prompt
            else:
                params["maskImage"] = mask_image

        elif task_type == "BACKGROUND_REMOVAL":
            params = {
                "image": background_removal_input
            }

        # Determine the actual task type to send to the backend
        backend_task_type = "TEXT_IMAGE" if task_type == "TEXT_IMAGE with conditioning" else task_type

        # Call image generator with prepared parameters
        output_path = image_generator.generate(
            task_type=backend_task_type,
            params=params,
            config=image_generation_config
        )

        progress(1.0, desc=f"{task_type} completed!")
        logger.info(f"Image generated successfully: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error during {task_type}: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

# Custom CSS for better styling
custom_css = """
    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .sidebar {
        background-color: #f7f7f7;
        border-right: 1px solid #e0e0e0;
        padding: 20px;
        height: 100%;
    }
    
    .main-content {
        padding: 20px;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .tab-nav {
        background-color: #ffffff;
        border-bottom: 2px solid #f0f0f0;
        padding: 10px 0;
    }
    
    .primary-button {
        background-color: #87CEEB;
        color: orange;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: bold;
    }
    
    .group-container {
        background: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }

    .advanced-options {
        border-top: 1px solid #e0e0e0;
        margin-top: 20px;
        padding-top: 20px;
    }
"""

# Create the Gradio interface
with gr.Blocks(title="Amazon-Nova-AIGC", css=custom_css) as demo:
    with gr.Row():
        # Left sidebar with instructions
        with gr.Column(scale=1, elem_classes="sidebar"):
            gr.Markdown("""
                ## Nova AI Generator
                Transform your ideas into stunning images and videos with AI.
                
                ### Image Generation
                Choose from multiple generation types:
                - Text to Image: Generate images from text descriptions
                - Text to Image with Conditioning: Use edge detection or segmentation to guide generation
                - Color Guided Generation: Create images with specific color palettes
                - Image Variation: Generate variations of existing images
                - Inpainting: Replace or modify specific areas in images
                - Outpainting: Extend images beyond their original boundaries
                - Background Removal: Remove backgrounds from images

                ### Image Analysis
                - Image to Prompt: Analyze images to generate detailed prompts
                
                ### Video Generation
                Create videos from:
                - Text descriptions: Generate videos from text prompts
                - Image transformations: Turn static images into dynamic videos
                - Program Maker: Combine multiple images into a sequence of videos
                
                ### Object Detection
                - Detect specific objects in images
                - Generate binary masks for detected objects
                - Support for both Nova Lite and Pro models
                
                ### Tips
                - Be specific in your descriptions
                - Include details about style and quality
                - Review optimized prompts before generating
                - Check [Nova's Prompting Guide](https://docs.aws.amazon.com/nova/latest/userguide/prompting-creation.html) for best practices
            """)
        
        # Main content area
        with gr.Column(scale=3, elem_classes="main-content"):
            with gr.Tabs():
                # New Image Generation Tab
                with gr.Tab("Image Generation"):
                    with gr.Group(elem_classes="group-container"):
                        task_type = gr.Dropdown(
                            choices=[
                                "TEXT_IMAGE",
                                "TEXT_IMAGE with conditioning",
                                "COLOR_GUIDED_GENERATION",
                                "IMAGE_VARIATION",
                                "INPAINTING",
                                "OUTPAINTING",
                                "BACKGROUND_REMOVAL"
                            ],
                            label="Task Type",
                            value="TEXT_IMAGE"
                        )
                        
                        # Common inputs
                        with gr.Group() as text_inputs:
                            text = gr.Textbox(
                                label="Prompt",
                                placeholder="Describe what you want to generate...",
                                lines=3
                            )
                            negative_text = gr.Textbox(
                                label="Negative Prompt",
                                placeholder="Describe what you want to avoid...",
                                lines=2,
                                value="bad quality, low resolution"
                            )
                            optimize_btn = gr.Button("✨ Optimize Prompt", elem_classes="primary-button")
                            optimized_text = gr.Textbox(
                                label="Optimized Prompt",
                                lines=3,
                                interactive=True
                            )
                        
                        # Separate input images for each task type
                        with gr.Group() as input_images_group:
                            conditioning_input = gr.Image(label="Input Image (Conditioning)", type="filepath", sources=["upload"], visible=False)
                            variation_input = gr.Image(label="Input Image (Variation)", type="filepath", sources=["upload"], visible=False)
                            inpainting_input = gr.Image(label="Input Image (Inpainting)", type="filepath", sources=["upload"], visible=False)
                            outpainting_input = gr.Image(label="Input Image (Outpainting)", type="filepath", sources=["upload"], visible=False)
                            background_removal_input = gr.Image(label="Input Image (Background Removal)", type="filepath", sources=["upload"], visible=False)
                        
                        # Conditioning controls
                        with gr.Group(visible=False) as conditioning_controls:
                            control_mode = gr.Radio(
                                choices=["CANNY_EDGE", "SEGMENTATION"],
                                label="Control Mode"
                            )
                            control_strength = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.7,
                                label="Control Strength"
                            )
                        
                        # Color guided controls
                        with gr.Group(visible=False) as color_controls:
                            colors = gr.Textbox(
                                label="Colors (comma-separated hex values)",
                                placeholder="#FF0000,#00FF00,#0000FF"
                            )
                            color_guided_reference = gr.Image(
                                label="Reference Image (Color Guided)",
                                type="filepath",
                                sources=["upload"]
                            )
                        
                        # Variation controls
                        with gr.Group(visible=False) as variation_controls:
                            similarity_strength = gr.Slider(
                                minimum=0.2,
                                maximum=1.0,
                                value=0.7,
                                label="Similarity Strength"
                            )
                        
                        # Inpainting/Outpainting controls
                        with gr.Group(visible=False) as painting_controls:
                            mask_prompt = gr.Textbox(
                                label="Mask Prompt",
                                placeholder="Describe the area to modify..."
                            )
                            mask_image = gr.Image(
                                label="Mask Image",
                                type="filepath",
                                sources=["upload"]
                            )
                            outpainting_mode = gr.Radio(
                                choices=["DEFAULT", "PRECISE"],
                                label="Outpainting Mode",
                                visible=False
                            )
                        
                        # Common configuration
                        with gr.Group() as advanced_options:
                            with gr.Accordion("Advanced Options", open=False):
                                with gr.Row():
                                    width = gr.Number(
                                        label="Width",
                                        value=1280,
                                        minimum=64,
                                        maximum=2048
                                    )
                                    height = gr.Number(
                                        label="Height",
                                        value=720,
                                        minimum=64,
                                        maximum=2048
                                    )
                                
                                quality = gr.Radio(
                                    choices=["standard", "premium"],
                                    label="Quality",
                                    value="standard"
                                )
                                
                                cfg_scale = gr.Slider(
                                    minimum=1.0,
                                    maximum=20.0,
                                    value=8.0,
                                    label="CFG Scale"
                                )
                                with gr.Row():
                                    seed = gr.Number(
                                        label="Seed (0 ~ 858,993,459)",
                                        value=0,
                                        minimum=0,
                                        maximum=858993459
                                    )
                                    random_seed_btn = gr.Button("🎲 Random", elem_classes="primary-button")
                                num_images = gr.Number(
                                    label="Number of Images",
                                    value=1,
                                    minimum=1,
                                    maximum=4
                                )
                        
                        generate_btn = gr.Button("🎨 Generate", elem_classes="primary-button")
                        
                        # Create separate output components for each task type
                        with gr.Group() as output_group:
                            text_image_output = gr.Gallery(label="Generated Images (Text to Image)", visible=True)
                            conditioned_image_output = gr.Gallery(label="Generated Images (Conditioned)", visible=False)
                            color_guided_output = gr.Gallery(label="Generated Images (Color Guided)", visible=False)
                            variation_output = gr.Gallery(label="Generated Images (Variation)", visible=False)
                            inpainting_output = gr.Gallery(label="Generated Images (Inpainting)", visible=False)
                            outpainting_output = gr.Gallery(label="Generated Images (Outpainting)", visible=False)
                            background_removal_output = gr.Gallery(label="Generated Images (Background Removal)", visible=False)
                        
                        def update_ui(task):
                            """Handle visibility of controls and outputs based on task type"""
                            text_visible = task != "BACKGROUND_REMOVAL"
                            conditioning_visible = task == "TEXT_IMAGE with conditioning"
                            color_visible = task == "COLOR_GUIDED_GENERATION"
                            variation_visible = task == "IMAGE_VARIATION"
                            painting_visible = task in ["INPAINTING", "OUTPAINTING"]
                            outpainting_visible = task == "OUTPAINTING"
                            advanced_visible = task != "BACKGROUND_REMOVAL"

                            # Input image visibility
                            input_visibility = {
                                "TEXT_IMAGE": [False, False, False, False, False],
                                "TEXT_IMAGE with conditioning": [True, False, False, False, False],
                                "COLOR_GUIDED_GENERATION": [False, False, False, False, False],
                                "IMAGE_VARIATION": [False, True, False, False, False],
                                "INPAINTING": [False, False, True, False, False],
                                "OUTPAINTING": [False, False, False, True, False],
                                "BACKGROUND_REMOVAL": [False, False, False, False, True]
                            }
                            current_input_visibility = input_visibility[task]
                            
                            # Update output visibility based on task
                            outputs_visibility = {
                                "TEXT_IMAGE": [True, False, False, False, False, False, False],
                                "TEXT_IMAGE with conditioning": [False, True, False, False, False, False, False],
                                "COLOR_GUIDED_GENERATION": [False, False, True, False, False, False, False],
                                "IMAGE_VARIATION": [False, False, False, True, False, False, False],
                                "INPAINTING": [False, False, False, False, True, False, False],
                                "OUTPAINTING": [False, False, False, False, False, True, False],
                                "BACKGROUND_REMOVAL": [False, False, False, False, False, False, True]
                            }
                            
                            current_visibility = outputs_visibility[task]
                            
                            return [
                                gr.update(visible=text_visible),
                                gr.update(visible=current_input_visibility[0]),  # conditioning
                                gr.update(visible=current_input_visibility[1]),  # variation
                                gr.update(visible=current_input_visibility[2]),  # inpainting
                                gr.update(visible=current_input_visibility[3]),  # outpainting
                                gr.update(visible=current_input_visibility[4]),  # background removal
                                gr.update(visible=conditioning_visible),
                                gr.update(visible=color_visible),
                                gr.update(visible=variation_visible),
                                gr.update(visible=painting_visible),
                                gr.update(visible=outpainting_visible),
                                gr.update(visible=advanced_visible),
                                gr.update(visible=current_visibility[0]),
                                gr.update(visible=current_visibility[1]),
                                gr.update(visible=current_visibility[2]),
                                gr.update(visible=current_visibility[3]),
                                gr.update(visible=current_visibility[4]),
                                gr.update(visible=current_visibility[5]),
                                gr.update(visible=current_visibility[6])
                            ]
                        
                        task_type.change(
                            fn=update_ui,
                            inputs=task_type,
                            outputs=[
                                text_inputs,
                                conditioning_input,
                                variation_input,
                                inpainting_input,
                                outpainting_input,
                                background_removal_input,
                                conditioning_controls,
                                color_controls,
                                variation_controls,
                                painting_controls,
                                outpainting_mode,
                                advanced_options,
                                text_image_output,
                                conditioned_image_output,
                                color_guided_output,
                                variation_output,
                                inpainting_output,
                                outpainting_output,
                                background_removal_output
                            ]
                        )
                        
                        # Connect optimize button
                        optimize_btn.click(
                            fn=optimize_image_prompt,
                            inputs=[text],
                            outputs=optimized_text
                        )
                        
                        def route_output(output_paths, task):
                            """Route the output to the correct component based on task type"""
                            outputs = [None] * 7  # Initialize all outputs as None
                            if not isinstance(output_paths, list):
                                output_paths = [output_paths]
                            task_index = {
                                "TEXT_IMAGE": 0,
                                "TEXT_IMAGE with conditioning": 1,
                                "COLOR_GUIDED_GENERATION": 2,
                                "IMAGE_VARIATION": 3,
                                "INPAINTING": 4,
                                "OUTPAINTING": 5,
                                "BACKGROUND_REMOVAL": 6
                            }
                            if task in task_index:
                                outputs[task_index[task]] = output_paths
                            return outputs

                        def generate_random_seed():
                            """Generate a random seed value"""
                            import random
                            return random.randint(0, 858993459)

                        # Connect random seed button
                        random_seed_btn.click(
                            fn=generate_random_seed,
                            inputs=[],
                            outputs=[seed]
                        )

                        # Connect generation button
                        generate_btn.click(
                            fn=lambda *args: route_output(handle_image_generation(*args), args[0]),
                            inputs=[
                                task_type,
                                optimized_text,  # Use optimized text instead of original
                                negative_text,
                                width,
                                height,
                                quality,
                                cfg_scale,
                                seed,
                                num_images,
                                conditioning_input,  # For TEXT_IMAGE with conditioning
                                variation_input,    # For IMAGE_VARIATION
                                inpainting_input,   # For INPAINTING
                                outpainting_input,  # For OUTPAINTING
                                background_removal_input,  # For BACKGROUND_REMOVAL
                                control_mode,
                                control_strength,
                                colors,
                                color_guided_reference,
                                similarity_strength,
                                mask_prompt,
                                mask_image,
                                outpainting_mode
                            ],
                            outputs=[
                                text_image_output,
                                conditioned_image_output,
                                color_guided_output,
                                variation_output,
                                inpainting_output,
                                outpainting_output,
                                background_removal_output
                            ]
                        )

                # Image to Prompt tab
                with gr.Tab("Image to Prompt"):
                    with gr.Group(elem_classes="group-container"):
                        img2prompt_input = gr.Image(
                            label="Upload Image to Analyze",
                            type="filepath",
                            sources=["upload"]
                        )
                        img2prompt_analyze_btn = gr.Button("🔍 Analyze Image", elem_classes="primary-button")
                        img2prompt_output = gr.Textbox(
                            label="Generated Prompt",
                            lines=3,
                            interactive=True
                        )

                # Text to Video tab
                with gr.Tab("Text to Video"):
                    with gr.Group(elem_classes="group-container"):
                        txt2vid_input = gr.Textbox(
                            label="Enter your creative prompt",
                            placeholder="Describe the video you want to generate...",
                            lines=3
                        )
                        txt2vid_optimize_btn = gr.Button("✨ Optimize Prompt", elem_classes="primary-button")
                        txt2vid_optimized = gr.Textbox(
                            label="Optimized Prompt",
                            lines=3,
                            interactive=True
                        )
                        txt2vid_generate_btn = gr.Button("🎬 Generate Video", elem_classes="primary-button")
                        txt2vid_output = gr.Video(label="Generated Video")

                # Object Detection and Masking tab
                with gr.Tab("Object Detection"):
                    with gr.Group(elem_classes="group-container"):
                        detect_image = gr.Image(
                            label="Upload Image",
                            type="filepath",
                            sources=["upload"]
                        )
                        detect_object = gr.Textbox(
                            label="Object to Detect",
                            placeholder="Enter the object name to detect (e.g., 'bottle', 'car', 'dog')"
                        )
                        with gr.Row():
                            detect_model = gr.Dropdown(
                                choices=["us.amazon.nova-lite-v1:0", "us.amazon.nova-pro-v1:0"],
                                value="us.amazon.nova-lite-v1:0",
                                label="Model"
                            )
                            detect_size = gr.Slider(
                                minimum=360,
                                maximum=1920,
                                value=480,
                                step=4,
                                label="Image Processing Size"
                            )
                        detect_btn = gr.Button("🎯 Detect and Create Mask", elem_classes="primary-button")
                        with gr.Row():
                            detect_output = gr.Image(label="Detection Result")
                            mask_output = gr.Image(label="Mask Result")
                        detect_coords = gr.JSON(label="Detection Coordinates")
                        mask_path = gr.Textbox(label="Saved Mask Path")

                # Image to Video tab
                with gr.Tab("Image to Video"):
                    with gr.Group(elem_classes="group-container"):
                        img2vid_input = gr.Image(
                            label="Upload Starting Image",
                            type="filepath",
                            sources=["upload"]
                        )
                        img2vid_prompt = gr.Textbox(
                            label="Describe your transformation",
                            placeholder="How would you like to transform this image?",
                            lines=3
                        )
                        img2vid_optimize_btn = gr.Button("✨ Optimize Prompt", elem_classes="primary-button")
                        img2vid_optimized = gr.Textbox(
                            label="Optimized Prompt",
                            lines=3,
                            interactive=True
                        )
                        img2vid_generate_btn = gr.Button("🎬 Generate Video", elem_classes="primary-button")
                        img2vid_output = gr.Video(label="Generated Video")
                
                # Program Maker tab
                with gr.Tab("Program Maker"):
                    with gr.Group(elem_classes="group-container"):
                        gr.Markdown("""
                        ## Program Maker
                        
                        Upload multiple images (up to 10) to create a sequence of videos that will be combined into a final video.
                        
                        1. Upload your images in the desired order
                        2. Generate video prompts for each image
                        3. Generate videos for each image
                        4. Preview the videos
                        5. Combine the videos into a final video
                        """)
                        
                        # Image upload section
                        with gr.Row():
                            program_images = gr.Gallery(
                                label="Uploaded Images",
                                show_label=True,
                                elem_id="program_images",
                                columns=5,
                                height="auto"
                            )
                        
                        with gr.Row():
                            program_image_upload = gr.File(
                                label="Upload Images (Max 10)",
                                file_count="multiple",
                                file_types=["image"],
                                elem_id="program_image_upload"
                            )
                            program_clear_btn = gr.Button("Clear Images", elem_classes="primary-button")
                        
                        # Prompt generation section
                        program_generate_prompts_btn = gr.Button("1️⃣ Generate Video Prompts", elem_classes="primary-button")
                        
                        program_prompts = gr.Dataframe(
                            headers=["Image", "Prompt"],
                            datatype=["str", "str"],
                            row_count=10,
                            col_count=(2, "fixed"),
                            interactive=True,
                            label="Generated Prompts (Editable)"
                        )
                        
                        # Video generation section
                        with gr.Row():
                            program_generate_videos_btn = gr.Button("2️⃣ Generate Videos", elem_classes="primary-button")
                            program_video_progress = gr.Textbox(
                                label="Progress",
                                value="Not started",
                                interactive=False
                            )
                        
                        # Video preview section
                        program_videos = gr.Gallery(
                            label="Generated Videos",
                            show_label=True,
                            elem_id="program_videos",
                            columns=5,
                            height="auto"
                        )
                        
                        # Video combination section
                        with gr.Row():
                            with gr.Column(scale=1):
                                program_transition = gr.Radio(
                                    choices=["fade", "none"],
                                    label="Transition Type",
                                    value="fade"
                                )
                            
                            with gr.Column(scale=2):
                                program_durations = gr.Textbox(
                                    label="Durations (seconds per clip, 1-6)",
                                    placeholder="e.g., 4,3,4,5,3",
                                    value="4"
                                )
                        
                        program_combine_btn = gr.Button("3️⃣ Combine Videos", elem_classes="primary-button")
                        program_final_video = gr.Video(label="Final Combined Video")

    # Connect the components for image to prompt
    img2prompt_analyze_btn.click(
        fn=analyze_image_to_prompt,
        inputs=[img2prompt_input],
        outputs=img2prompt_output
    )

    # Connect the components for existing tabs
    txt2vid_optimize_btn.click(
        fn=optimize_video_prompt,
        inputs=[txt2vid_input],
        outputs=txt2vid_optimized
    )
    
    txt2vid_generate_btn.click(
        fn=generate_video,
        inputs=[txt2vid_optimized],
        outputs=txt2vid_output
    )
    
    img2vid_optimize_btn.click(
        fn=optimize_video_prompt,
        inputs=[img2vid_prompt, img2vid_input],
        outputs=img2vid_optimized
    )
    
    img2vid_generate_btn.click(
        fn=generate_video,
        inputs=[img2vid_optimized, img2vid_input],
        outputs=img2vid_output
    )

    # Connect object detection components
    detect_btn.click(
        fn=detect_and_mask,
        inputs=[detect_image, detect_object, detect_model, detect_size],
        outputs=[detect_output, mask_output, detect_coords, mask_path]
    )
    
    # Program Maker functions
    def upload_images(files):
        """Process uploaded image files and return their paths"""
        if not files:
            return []
        
        # Limit to 10 images
        files = files[:10]
        
        # Get file paths
        image_paths = [file.name for file in files]
        
        # Return both paths for display and actual file paths
        return image_paths
    
    def clear_images():
        """Clear all uploaded images"""
        return None, None, None, "Not started", None
    
    def generate_video_prompts(image_paths):
        """Generate video prompts for the uploaded images"""
        if not image_paths:
            return [], "No images uploaded"
        
        try:
            # Process image paths - handle both string paths and tuples (path, None)
            processed_paths = []
            for item in image_paths:
                if isinstance(item, tuple) and len(item) == 2:
                    # Extract just the path from (path, None) tuple
                    processed_paths.append(item[0])
                elif isinstance(item, str):
                    processed_paths.append(item)
                else:
                    logger.warning(f"Skipping invalid image path format: {item}")
            
            if not processed_paths:
                return [], "No valid image paths found"
            
            # Generate prompts
            prompts_dict = image_to_video_prompt.generate_prompts(processed_paths)
            
            # Convert to dataframe format
            prompts_data = [[path, prompts_dict[path]] for path in processed_paths]
            
            return prompts_data, "Prompts generated successfully"
        except Exception as e:
            logger.error(f"Error generating prompts: {str(e)}")
            return [], f"Error: {str(e)}"
    
    def generate_videos_from_prompts(image_paths, prompts_data, progress=gr.Progress()):
        """Generate videos for each image using the provided prompts concurrently"""
        # Check if image_paths is empty
        if not image_paths:
            return None, "No images available"
        
        # Check if prompts_data is empty (handle DataFrame properly)
        try:
            import pandas as pd
            if isinstance(prompts_data, pd.DataFrame):
                if prompts_data.empty:
                    return None, "No prompts available"
            elif not prompts_data:  # Handle other types (list, etc.)
                return None, "No prompts available"
        except Exception as e:
            logger.warning(f"Error checking prompts_data: {str(e)}")
            if not prompts_data:  # Fallback check
                return None, "No prompts available"
        
        try:
            # Import ThreadPoolExecutor for concurrent processing
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import threading
            
            # Process image paths - handle both string paths and tuples (path, None)
            processed_paths = []
            for item in image_paths:
                if isinstance(item, tuple) and len(item) == 2:
                    # Extract just the path from (path, None) tuple
                    processed_paths.append(item[0])
                elif isinstance(item, str):
                    processed_paths.append(item)
                else:
                    logger.warning(f"Skipping invalid image path format: {item}")
            
            if not processed_paths:
                return None, "No valid image paths found"
            
            # Convert prompts data to dictionary
            prompts = {}
            try:
                import pandas as pd
                if isinstance(prompts_data, pd.DataFrame):
                    # Handle pandas DataFrame
                    for index, row in prompts_data.iterrows():
                        if len(row) >= 2:
                            prompts[row[0]] = row[1]
                else:
                    # Handle list or other iterable
                    for row in prompts_data:
                        if len(row) >= 2:
                            # The key in prompts_data might be the processed path
                            prompts[row[0]] = row[1]
            except Exception as e:
                logger.error(f"Error processing prompts data: {str(e)}")
                return None, f"Error processing prompts: {str(e)}"
            
            # Initialize image resizer
            resizer = IntelligentImageResizer()
            
            # Dictionary to store processed images
            processed_images = {}
            
            # Process all images first (resize them)
            logger.info("Processing images for video generation...")
            for i, image_path in enumerate(processed_paths):
                if image_path in prompts:
                    progress((i * 0.2) / len(processed_paths), f"Processing image {i+1}/{len(processed_paths)}")
                    
                    try:
                        # Create output path for processed image
                        directory = os.path.dirname(image_path)
                        filename = os.path.basename(image_path)
                        base_name, ext = os.path.splitext(filename)
                        processed_image_path = os.path.join(directory, f"{base_name}_processed{ext}")
                        
                        # Resize image
                        processed_image = resizer.resize_image(
                            image_path=image_path,
                            output_path=processed_image_path,
                            target_width=1280,
                            target_height=720
                        )
                        
                        logger.info(f"Image processed successfully: {processed_image}")
                        processed_images[image_path] = processed_image
                    except Exception as e:
                        logger.error(f"Error processing image {image_path}: {str(e)}")
            
            # Function to generate a video for a single image
            def generate_single_video(idx, image_path, current_batchid, current_batch_dir):
                if image_path not in prompts or image_path not in processed_images:
                    return idx, None
                
                try:
                    logger.info(f"Generating video for image {idx+1}: {image_path}")
                    processed_image = processed_images[image_path]
                    
                    # Generate video with processed image
                    response = video_generator.generate_video(
                        text=prompts[image_path], 
                        input_image_path=processed_image
                    )
                    
                    if "invocationArn" not in response:
                        logger.error(f"Failed to get invocation ARN for image {idx+1}")
                        return idx, None
                    
                    # Wait for completion
                    final_status = video_generator.wait_for_completion(response["invocationArn"])
                    
                    if final_status["status"] == "Completed":
                        # Get the base filename from the image path
                        image_basename = os.path.splitext(os.path.basename(image_path))[0]
                        # Create a new filename with batchid prefix
                        new_filename = f"{current_batchid}_{image_basename}.mp4"
                        # Set the download path to the batch directory
                        download_path = os.path.join(current_batch_dir, new_filename)
                        
                        # Download video with custom path
                        video_path = video_generator.download_video(
                            final_status["video_uri"], 
                            is_text_to_video=False,
                            custom_path=download_path
                        )
                        logger.info(f"Video generated successfully for image {idx+1}: {video_path}")
                        return idx, video_path
                    else:
                        logger.error(f"Video generation failed for image {idx+1}: {final_status.get('failure_message', 'Unknown error')}")
                        return idx, None
                except Exception as e:
                    logger.error(f"Error generating video for image {idx+1}: {str(e)}")
                    return idx, None
            
            # Track progress with a shared counter and lock
            completed_count = 0
            progress_lock = threading.Lock()
            
            def update_progress():
                nonlocal completed_count
                with progress_lock:
                    completed_count += 1
                    progress_val = 0.2 + (completed_count * 0.8 / len(processed_paths))
                    progress(progress_val, f"Generated {completed_count}/{len(processed_paths)} videos")
            
            # Generate a batchid for this set of videos
            batchid = video_maker._generate_batchid()
            
            # Create batch directory
            output_dir = os.getenv('VIDEO_OUTPUT_DIR', './output/generated_videos')
            batch_dir = os.path.join(output_dir, batchid)
            os.makedirs(batch_dir, exist_ok=True)
            
            # Save prompts to CSV file
            csv_path = os.path.join(batch_dir, f"{batchid}_prompts.csv")
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Image', 'Prompt'])
                for img_path, prompt in prompts.items():
                    csv_writer.writerow([img_path, prompt])
            logger.info(f"Saved prompts to CSV: {csv_path}")
            
            # Generate videos concurrently
            logger.info(f"Starting concurrent video generation for {len(processed_paths)} images")
            progress(0.2, f"Starting video generation (0/{len(processed_paths)})")
            
            # Dictionary to store results in order
            ordered_results = {}
            
            # Use ThreadPoolExecutor for concurrent processing
            with ThreadPoolExecutor(max_workers=10) as executor:
                # Submit all tasks
                future_to_idx = {
                    executor.submit(generate_single_video, i, path, batchid, batch_dir): i 
                    for i, path in enumerate(processed_paths) 
                    if path in prompts and path in processed_images
                }
                
                # Process results as they complete
                for future in as_completed(future_to_idx):
                    idx, video_path = future.result()
                    if video_path:
                        ordered_results[idx] = video_path
                    update_progress()
            
            # Collect results in original order
            video_paths = []
            for i in range(len(processed_paths)):
                if i in ordered_results:
                    video_paths.append(ordered_results[i])
            
            logger.info(f"Completed video generation. Generated {len(video_paths)} videos.")
            return video_paths, f"Generated {len(video_paths)} videos"
        except Exception as e:
            logger.error(f"Error generating videos: {str(e)}")
            return None, f"Error: {str(e)}"
    
    def combine_videos(video_paths, transition_type, durations_str):
        """Combine the generated videos into a final video"""
        if not video_paths:
            return None, "No videos to combine"
        
        try:
            # Process video paths - handle both string paths and tuples (path, None)
            processed_paths = []
            for item in video_paths:
                if isinstance(item, tuple) and len(item) == 2:
                    # Extract just the path from (path, None) tuple
                    processed_paths.append(item[0])
                elif isinstance(item, str):
                    processed_paths.append(item)
                else:
                    logger.warning(f"Skipping invalid video path format: {item}")
            
            if not processed_paths:
                return None, "No valid video paths found"
            
            # Parse durations
            try:
                if "," in durations_str:
                    durations = [int(d.strip()) for d in durations_str.split(",")]
                else:
                    # Use the same duration for all videos
                    durations = [int(durations_str.strip())] * len(processed_paths)
            except ValueError:
                # Default to 4 seconds if parsing fails
                durations = [4] * len(processed_paths)
            
            # Ensure we have the right number of durations
            if len(durations) < len(processed_paths):
                durations.extend([4] * (len(processed_paths) - len(durations)))
            elif len(durations) > len(processed_paths):
                durations = durations[:len(processed_paths)]
            
            # Ensure all durations are between 1 and 6
            durations = [max(1, min(d, 6)) for d in durations]
            
            # Extract batchid from the first video path if possible
            # Format is typically: /path/to/batchid_directory/batchid_filename.mp4
            batchid = None
            if processed_paths:
                first_video = processed_paths[0]
                video_dir = os.path.dirname(first_video)
                video_filename = os.path.basename(first_video)
                
                # Try to extract batchid from directory name
                dir_batchid = os.path.basename(video_dir)
                if '_' in dir_batchid and len(dir_batchid.split('_')) == 2:
                    batchid = dir_batchid
                # If not found in directory, try from filename
                elif '_' in video_filename:
                    parts = video_filename.split('_')
                    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                        batchid = f"{parts[0]}_{parts[1]}"
            
            # If we couldn't extract batchid, generate a new one
            if not batchid:
                batchid = video_maker._generate_batchid()
            
            # Create batch directory
            output_dir = os.getenv('VIDEO_OUTPUT_DIR', './output/generated_videos')
            batch_dir = os.path.join(output_dir, batchid)
            os.makedirs(batch_dir, exist_ok=True)
            
            # Create output path with batchid
            output_path = os.path.join(batch_dir, f"{batchid}_combined.mp4")
            
            # Combine videos
            final_path = video_maker.create_video(
                video_paths=processed_paths,
                durations=durations,
                transition_type=transition_type,
                output_path=output_path
            )
            
            return final_path, f"Videos combined successfully: {final_path}"
        except Exception as e:
            logger.error(f"Error combining videos: {str(e)}")
            return None, f"Error: {str(e)}"
    
    # Connect Program Maker components
    program_image_upload.change(
        fn=upload_images,
        inputs=[program_image_upload],
        outputs=[program_images]
    )
    
    program_clear_btn.click(
        fn=clear_images,
        inputs=[],
        outputs=[program_images, program_prompts, program_videos, program_video_progress, program_final_video]
    )
    
    program_generate_prompts_btn.click(
        fn=generate_video_prompts,
        inputs=[program_images],
        outputs=[program_prompts, program_video_progress]
    )
    
    program_generate_videos_btn.click(
        fn=generate_videos_from_prompts,
        inputs=[program_images, program_prompts],
        outputs=[program_videos, program_video_progress]
    )
    
    program_combine_btn.click(
        fn=combine_videos,
        inputs=[program_videos, program_transition, program_durations],
        outputs=[program_final_video, program_video_progress]
    )

    # Launch the app
    if __name__ == "__main__":
        logger.info("Starting Gradio application...")
        demo.launch(server_name='0.0.0.0')
        logger.info("Gradio application stopped.")
