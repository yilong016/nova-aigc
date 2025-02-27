import moviepy
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.video.fx.all import fadeout, fadein
import os
import argparse
from tqdm import tqdm

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
            
            clip = VideoFileClip(path)
            clip = clip.subclip(0, duration)
            clip = self.transitions[transition_type](clip)
            clips.append(clip)
        
        # Combine all clips
        final_clip = concatenate_videoclips(clips, method="compose")
        
        # Save the result
        print("\nProcessing video...")
        final_clip.write_videofile(output_path, codec='libx264')
        
        # Clean up
        for clip in clips:
            clip.close()
        final_clip.close()
        
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
