import moviepy
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.video.fx.all import fadeout, fadein
import os
import argparse
from tqdm import tqdm
import logging
import time
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class VideoMaker:
    def __init__(self):
        self.transitions = {
            'fade': self._apply_fade_transition,
            'none': self._apply_no_transition
        }
    
    def _apply_fade_transition(self, clip, duration=1):
        """Apply fade in and fade out effects to a clip"""
        return fadein(fadeout(clip, duration), duration)
    
    def _apply_no_transition(self, clip, duration=1):
        """Return clip without any transition"""
        return clip
    
    def create_video_from_images(self, image_paths: List[str], video_generator, prompt_generator, 
                               durations: Optional[List[int]] = None, 
                               transition_type: str = 'fade',
                               output_path: str = 'output_video.mp4',
                               max_workers: int = 3) -> Tuple[str, List[str]]:
        """
        Create a video by generating individual videos from images and then combining them
        
        Args:
            image_paths (List[str]): List of paths to image files
            video_generator: NovaVideoGenerator instance
            prompt_generator: ImageToVideoPrompt instance
            durations (List[int], optional): List of durations for each clip (1-6 seconds)
                                            If None, defaults to 4 seconds per clip
            transition_type (str): Type of transition ('fade', or 'none')
            output_path (str): Path for the output video file
            max_workers (int): Maximum number of concurrent video generation tasks
        
        Returns:
            Tuple[str, List[str]]: Path to the output video file and list of individual video paths
        """
        logger.info(f"Starting video creation from {len(image_paths)} images")
        
        if not image_paths:
            raise ValueError("No image paths provided")
            
        if len(image_paths) > 10:
            raise ValueError("Maximum of 10 images allowed")
        
        # Set default durations if not provided
        if durations is None:
            durations = [4] * len(image_paths)
        
        if len(image_paths) != len(durations):
            raise ValueError("Number of images and durations must match")
        
        if not all(1 <= d <= 6 for d in durations):
            raise ValueError("All durations must be between 1 and 6 seconds")
        
        # Generate prompts for all images
        logger.info("Generating prompts for images...")
        try:
            prompts = prompt_generator.generate_prompts(image_paths)
            logger.info(f"Successfully generated {len(prompts)} prompts")
        except Exception as e:
            logger.error(f"Error generating prompts: {str(e)}")
            raise
        
        # Generate videos for each image
        logger.info("Generating videos from images...")
        video_paths = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all video generation tasks
            future_to_image = {
                executor.submit(
                    self._generate_video_for_image, 
                    image_path, 
                    prompts[image_path], 
                    video_generator
                ): image_path for image_path in image_paths
            }
            
            # Process results as they complete
            for future in tqdm(as_completed(future_to_image), total=len(image_paths), desc="Generating videos"):
                image_path = future_to_image[future]
                try:
                    video_path = future.result()
                    video_paths.append(video_path)
                    logger.info(f"Generated video for {image_path}: {video_path}")
                except Exception as e:
                    logger.error(f"Error generating video for {image_path}: {str(e)}")
                    raise
        
        # Sort video paths to match the original image order
        video_paths_sorted = []
        for image_path in image_paths:
            for video_path in video_paths:
                if os.path.basename(video_path).startswith(os.path.splitext(os.path.basename(image_path))[0]):
                    video_paths_sorted.append(video_path)
                    break
        
        if len(video_paths_sorted) != len(image_paths):
            logger.warning("Could not match all videos to images. Using unsorted video list.")
            video_paths_sorted = video_paths
        
        # Combine videos
        logger.info("Combining individual videos...")
        final_video_path = self.create_video(video_paths_sorted, durations, transition_type, output_path)
        
        return final_video_path, video_paths_sorted
    
    def _generate_video_for_image(self, image_path: str, prompt: str, video_generator) -> str:
        """
        Generate a video for a single image using the provided prompt
        
        Args:
            image_path (str): Path to the image file
            prompt (str): Prompt for video generation
            video_generator: NovaVideoGenerator instance
            
        Returns:
            str: Path to the generated video file
        """
        logger.info(f"Generating video for image: {image_path}")
        logger.info(f"Using prompt: {prompt}")
        
        try:
            # Start the video generation job
            response = video_generator.generate_video(text=prompt, input_image_path=image_path)
            
            if "invocationArn" not in response:
                raise ValueError("Failed to get invocation ARN from response")
                
            invocation_arn = response["invocationArn"]
            logger.info(f"Generation job started with ARN: {invocation_arn}")
            
            # Wait for the job to complete
            final_status = video_generator.wait_for_completion(invocation_arn)
            
            if final_status["status"] == "Failed":
                error_msg = final_status.get('failure_message', 'Unknown error')
                raise ValueError(f"Video generation failed: {error_msg}")
                
            if final_status["status"] == "Completed":
                video_uri = final_status["video_uri"]
                logger.info(f"Video generation completed. Downloading from {video_uri}")
                
                # Download the video
                video_path = video_generator.download_video(video_uri, is_text_to_video=False)
                logger.info(f"Video downloaded to: {video_path}")
                
                return video_path
            
            raise ValueError(f"Unexpected job status: {final_status['status']}")
            
        except Exception as e:
            logger.error(f"Error generating video: {str(e)}")
            raise
    
    def create_video(self, video_paths, durations, transition_type='fade', output_path='output_video.mp4'):
        """
        Create a video by combining multiple clips with transitions
        
        Args:
            video_paths (list): List of paths to video files
            durations (list): List of durations for each clip (1-6 seconds)
            transition_type (str): Type of transition ('fade', or 'none')
            output_path (str): Path for the output video file
        
        Returns:
            str: Path to the output video file
        """
        if len(video_paths) != len(durations):
            raise ValueError("Number of videos and durations must match")
        
        if not all(1 <= d <= 6 for d in durations):
            raise ValueError("All durations must be between 1 and 6 seconds")
        
        if transition_type not in self.transitions:
            raise ValueError(f"Unsupported transition type: {transition_type}")
        
        # Load and trim clips
        clips = []
        for path, duration in zip(video_paths, durations):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Video file not found: {path}")
            
            try:
                clip = VideoFileClip(path)
                # Trim if the clip is longer than the specified duration
                if clip.duration > duration:
                    clip = clip.subclip(0, duration)
                clip = self.transitions[transition_type](clip)
                clips.append(clip)
                logger.debug(f"Processed clip: {path}, duration: {duration}s")
            except Exception as e:
                logger.error(f"Error processing clip {path}: {str(e)}")
                raise
        
        # Combine all clips
        logger.info(f"Combining {len(clips)} clips...")
        final_clip = concatenate_videoclips(clips, method="compose")
        
        # Save the result
        logger.info(f"Processing final video to {output_path}...")
        final_clip.write_videofile(output_path, codec='libx264')
        
        # Clean up
        for clip in clips:
            clip.close()
        final_clip.close()
        
        logger.info(f"Video creation completed: {output_path}")
        return output_path

def main():
    parser = argparse.ArgumentParser(description='Combine video clips with transitions')
    parser.add_argument('--videos', nargs='+', required=True,
                      help='List of input video file paths (e.g., video1.mp4 video2.mp4)')
    parser.add_argument('--durations', type=int, nargs='+', required=True,
                      help='Duration in seconds for each clip, between 1-6 seconds (e.g., 3 4 2)')
    parser.add_argument('--transition', choices=['fade', 'none'],
                      default='fade', 
                      help='Transition effect type between clips:\n'
                           '  - fade: Smooth fade out/fade in transition\n'
                           '  - none: Direct cut without transition')
    parser.add_argument('--output', type=str, default='output_video.mp4',
                      help='Output video file path (default: output_video.mp4)')
    
    args = parser.parse_args()
    
    maker = VideoMaker()
    
    try:
        output_path = maker.create_video(args.videos, args.durations, args.transition, args.output)
        print(f"\nVideo created successfully: {output_path}")
    except Exception as e:
        print(f"\nError creating video: {str(e)}")

if __name__ == "__main__":
    main()


"""
Example usage for different transition types:

1. Fade transition (fade out/fade in):
python program_maker.py \
  --videos \
  ../output/sucai/1.mp4 \
  ../output/sucai/2.mp4 \
  ../output/sucai/3.mp4 \
    ../output/sucai/4.mp4 \
        ../output/sucai/5.mp4 \
            ../output/sucai/6.mp4 \
                ../output/sucai/7.mp4 \
                    ../output/sucai/8.mp4 \
                    ../output/sucai/9.mp4 \
  --durations 4 3 4 5 3 4 4 5 3 \
  --transition fade \
  --output ../output/program_video_1.mp4

2. No transition (direct cut):
python program_maker.py \
  --videos clip1.mp4 clip2.mp4 clip3.mp4 \
  --durations 3 4 2 \
  --transition none \
  --output direct_cut_video.mp4
"""
