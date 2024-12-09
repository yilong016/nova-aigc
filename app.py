import gradio as gr
import os
import logging
from video_generator import NovaVideoGenerator
from image_generator import NovaImageGenerator
from prompt_optimizer import PromptOptimizer, CanvasPromptOptimizer
import time
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

# Initialize the generators
logger.info("Initializing generators and optimizers...")
video_generator = NovaVideoGenerator()  # Will use S3 bucket from env vars
image_generator = NovaImageGenerator()
video_prompt_optimizer = PromptOptimizer()
image_prompt_optimizer = CanvasPromptOptimizer()

def optimize_video_prompt(text, image=None):
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

def optimize_image_prompt(text):
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

def generate_video(text, image=None, progress=gr.Progress()):
    """Generate video from text or image+text"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        logger.info(f"Starting video generation at {timestamp}")
        logger.info(f"Using prompt: {text}")
        if image:
            logger.info(f"Using input image: {image}")

        progress(0.05, desc="Starting video generation...")
        response = video_generator.generate_video(text=text, input_image_path=image)
        
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

def generate_image(text, progress=gr.Progress()):
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

def generate_outpainting(image, prompt, mask_prompt, progress=gr.Progress()):
    """Generate outpainted image"""
    try:
        logger.info("Starting outpainting generation")
        logger.info(f"Using prompt: {prompt}")
        logger.info(f"Using mask prompt: {mask_prompt}")
        logger.info(f"Using source image: {image}")
        
        progress(0.2, desc="Generating outpainting...")
        
        # Get dimensions from environment variables or use defaults
        dimensions = os.getenv('VIDEO_DEFAULT_DIMENSION', '1280x720').split('x')
        width = int(dimensions[0])
        height = int(dimensions[1])
        
        output_path = image_generator.outpainting(
            image_path=image,
            prompt=prompt,
            mask_prompt=mask_prompt,
            width=width,
            height=height
        )
        
        progress(1.0, desc="Outpainting completed!")
        logger.info(f"Outpainting generated successfully: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error during outpainting: {str(e)}", exc_info=True)
        return f"Error generating outpainting: {str(e)}"

# Custom CSS for better styling
custom_css = """
    /* Global styles */
    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sidebar styling */
    .sidebar {
        background-color: #f7f7f7;
        border-right: 1px solid #e0e0e0;
        padding: 20px;
        height: 100%;
    }
    
    .sidebar h2 {
        color: #2a2a2a;
        font-size: 1.5em;
        margin-bottom: 1em;
        font-weight: 600;
    }
    
    .sidebar h3 {
        color: #4a4a4a;
        font-size: 1.1em;
        margin-top: 1.5em;
        margin-bottom: 0.5em;
        font-weight: 500;
    }
    
    .sidebar p {
        color: #666;
        font-size: 0.95em;
        line-height: 1.5;
        margin-bottom: 1em;
    }
    
    /* Main content styling */
    .main-content {
        padding: 20px;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Tab styling */
    .tab-nav {
        background-color: #ffffff;
        border-bottom: 2px solid #f0f0f0;
        padding: 10px 0;
    }
    
    /* Button styling with blue theme */
    .primary-button {
        background-color: #87CEEB;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0, 102, 204, 0.2);
    }
    
    .primary-button:hover {
        background-color: #0052a3;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 102, 204, 0.3);
    }
    
    .primary-button:active {
        background-color: #004080;
        transform: translateY(0);
        box-shadow: 0 1px 2px rgba(0, 102, 204, 0.2);
    }
    
    /* Input styling */
    .input-box {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 12px;
        margin-bottom: 15px;
        background-color: #ffffff;
    }
    
    /* Output styling */
    .output-display {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    /* Group container styling */
    .group-container {
        background: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }

    /* Progress bar styling */
    .progress-bar {
        height: 8px;
        background-color: #e0e0e0;
        border-radius: 4px;
        overflow: hidden;
    }

    .progress-bar-fill {
        height: 100%;
        background-color: #0066cc;
        transition: width 0.5s ease;
    }
"""

# Create the Gradio interface
logger.info("Setting up Gradio interface...")
with gr.Blocks(css=custom_css) as demo:
    with gr.Row():
        # Left sidebar with instructions
        with gr.Column(scale=1, elem_classes="sidebar"):
            gr.Markdown("""
                ## Nova AI Generator
                Transform your ideas into stunning images and videos with AI.
                
                ### Text to Image
                1. Enter your creative prompt
                2. Click 'Optimize Prompt' to enhance it
                3. Generate your image
                
                ### Image to Image (Outpainting)
                1. Upload your source image
                2. Describe the area to modify (mask prompt)
                3. Describe what to add or change
                4. Generate your enhanced image
                
                ### Text to Video
                1. Enter your creative prompt
                2. Click 'Optimize Prompt' to enhance it
                3. Generate your video
                
                ### Image to Video
                1. Upload your starting image
                2. Describe your desired transformation
                3. Create your video
                
                ### Tips
                - Be specific in your descriptions
                - Include details about style and quality
                - For outpainting, clearly describe which area to modify
                - Review optimized prompts before generating
                
                ### Learn More
                For detailed guidance on creating effective prompts, check out the [Nova Prompt Best Practices Guide](https://docs.aws.amazon.com/nova/latest/userguide/prompting-creation.html)
            """)
        
        # Main content area
        with gr.Column(scale=3, elem_classes="main-content"):
            with gr.Tabs():
                # Text to Image tab
                with gr.Tab("Text to Image"):
                    with gr.Group(elem_classes="group-container"):
                        txt2img_input = gr.Textbox(
                            label="Enter your creative prompt",
                            placeholder="Describe the image you want to generate...",
                            lines=3,
                            elem_classes="input-box"
                        )
                        
                        txt2img_optimize_btn = gr.Button("✨ Optimize Prompt", elem_classes="primary-button")
                        
                        txt2img_optimized = gr.Textbox(
                            label="Optimized Prompt",
                            lines=3,
                            interactive=True,
                            elem_classes="input-box"
                        )
                        
                        txt2img_generate_btn = gr.Button("🎨 Generate Image", elem_classes="primary-button")
                        
                        txt2img_output = gr.Image(
                            label="Generated Image",
                            elem_classes="output-display"
                        )
                
                # Image to Image tab
                with gr.Tab("Image to Image"):
                    with gr.Group(elem_classes="group-container"):
                        img2img_input = gr.Image(
                            label="Upload Source Image",
                            type="filepath"
                        )
                        
                        img2img_mask_prompt = gr.Textbox(
                            label="Describe Area to Keep",
                            placeholder="Describe which part of the image to keep (e.g., 'coffee machine', 'the sky area', 'the bottom portion')",
                            lines=2,
                            elem_classes="input-box"
                        )
                        
                        img2img_prompt = gr.Textbox(
                            label="Describe What to Add/Change",
                            placeholder="What would you like to add or change in the selected area?",
                            lines=3,
                            elem_classes="input-box"
                        )
                        
                        img2img_optimize_btn = gr.Button("✨ Optimize Prompt", elem_classes="primary-button")
                        
                        img2img_optimized = gr.Textbox(
                            label="Optimized Prompt",
                            lines=3,
                            interactive=True,
                            elem_classes="input-box"
                        )
                        
                        img2img_generate_btn = gr.Button("🎨 Generate Outpainting", elem_classes="primary-button")
                        
                        img2img_output = gr.Image(
                            label="Generated Image",
                            elem_classes="output-display"
                        )
                
                # Text to Video tab
                with gr.Tab("Text to Video"):
                    with gr.Group(elem_classes="group-container"):
                        txt2vid_input = gr.Textbox(
                            label="Enter your creative prompt",
                            placeholder="Describe the video you want to generate...",
                            lines=3,
                            elem_classes="input-box"
                        )
                        
                        txt2vid_optimize_btn = gr.Button("✨ Optimize Prompt", elem_classes="primary-button")
                        
                        txt2vid_optimized = gr.Textbox(
                            label="Optimized Prompt",
                            lines=3,
                            interactive=True,
                            elem_classes="input-box"
                        )
                        
                        txt2vid_generate_btn = gr.Button("🎬 Generate Video", elem_classes="primary-button")
                        
                        txt2vid_output = gr.Video(
                            label="Generated Video",
                            elem_classes="output-display"
                        )
                
                # Image to Video tab
                with gr.Tab("Image to Video"):
                    with gr.Group(elem_classes="group-container"):
                        img2vid_input = gr.Image(
                            label="Upload Starting Image",
                            type="filepath"
                        )
                        
                        img2vid_prompt = gr.Textbox(
                            label="Describe your transformation",
                            placeholder="How would you like to transform this image?",
                            lines=3,
                            elem_classes="input-box"
                        )
                        
                        img2vid_optimize_btn = gr.Button("✨ Optimize Prompt", elem_classes="primary-button")
                        
                        img2vid_optimized = gr.Textbox(
                            label="Optimized Prompt",
                            lines=3,
                            interactive=True,
                            elem_classes="input-box"
                        )
                        
                        img2vid_generate_btn = gr.Button("🎬 Generate Video", elem_classes="primary-button")
                        
                        img2vid_output = gr.Video(
                            label="Generated Video",
                            elem_classes="output-display"
                        )
    
    # Connect the components
    # Text to Image
    txt2img_optimize_btn.click(
        fn=optimize_image_prompt,
        inputs=[txt2img_input],
        outputs=txt2img_optimized
    )
    
    txt2img_generate_btn.click(
        fn=generate_image,
        inputs=[txt2img_optimized],
        outputs=txt2img_output
    )
    
    # Image to Image
    img2img_optimize_btn.click(
        fn=optimize_image_prompt,
        inputs=[img2img_prompt],
        outputs=img2img_optimized
    )
    
    img2img_generate_btn.click(
        fn=generate_outpainting,
        inputs=[img2img_input, img2img_optimized, img2img_mask_prompt],
        outputs=img2img_output
    )
    
    # Text to Video
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
    
    # Image to Video
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

# Launch the app
if __name__ == "__main__":
    logger.info("Starting Gradio application...")
    demo.launch()
    logger.info("Gradio application stopped.")
