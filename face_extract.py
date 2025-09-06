import os
import cv2
import numpy as np
import tempfile
import argparse
import hashlib
import random
import shutil
import yt_dlp
import fal_client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up fal.ai API key from .env file
if 'FAL_API_KEY' in os.environ and 'FAL_KEY' not in os.environ:
    os.environ['FAL_KEY'] = os.environ['FAL_API_KEY']

# ---------- YouTube download helper ----------
def get_cache_path(url):
    """Generate a cache file path based on the URL hash."""
    cache_dir = os.path.expanduser("~/.cache/face_extract")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create hash of the URL for unique filename
    url_hash = hashlib.md5(url.encode()).hexdigest()
    return os.path.join(cache_dir, f"video_{url_hash}.mp4")

def download_youtube_video(url, use_cache=True):
    """Download a YouTube video and return the path to the downloaded file."""
    if use_cache:
        cache_path = get_cache_path(url)
        if os.path.exists(cache_path):
            return cache_path, False  # False means not temporary
    
    try:
        if use_cache:
            # Download directly to cache path
            output_path = get_cache_path(url)
            ydl_opts = {
                'format': 'best[ext=mp4]/best',  # Try mp4 first, fallback to best
                'outtmpl': output_path,
                'quiet': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            return output_path, False  # Cached, not temporary
            
        else:
            # Download to temporary directory
            temp_dir = tempfile.mkdtemp()
            ydl_opts = {
                'format': 'best[ext=mp4]/best',  # Try mp4 first, fallback to best
                'outtmpl': os.path.join(temp_dir, 'video.%(ext)s'),
                'quiet': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url)
                # Get the actual downloaded filename
                ext = info.get('ext', 'mp4')
                output_path = os.path.join(temp_dir, f'video.{ext}')
            
            return output_path, True  # Temporary
            
    except Exception as e:
        raise RuntimeError(f"Failed to download YouTube video: {str(e)}")

# ---------- Scene generation with nano-banana edit ----------
def generate_scenes_with_ai_host(frame_paths, ai_host_path, output_dir):
    """Generate new scenes using nano-banana edit with AI host in frame settings."""
    try:
        # Upload all images to fal.ai
        print("Uploading images to fal.ai...")
        
        # Upload frame image (only 1 frame now)
        frame_urls = []
        for i, frame_path in enumerate(frame_paths):
            uploaded_frame = fal_client.upload_file(frame_path)
            frame_url = uploaded_frame if isinstance(uploaded_frame, str) else uploaded_frame.url
            frame_urls.append(frame_url)
            print(f"  Uploaded frame: {frame_url}")
        
        # Upload AI host image
        uploaded_host = fal_client.upload_file(ai_host_path)
        host_url = uploaded_host if isinstance(uploaded_host, str) else uploaded_host.url
        print(f"  Uploaded AI host: {host_url}")
        
        # Create all image URLs list (frames + host)
        all_image_urls = frame_urls + [host_url]
        
        # Queue update handler for logs
        def on_queue_update(update):
            if isinstance(update, fal_client.InProgress):
                for log in update.logs:
                    print(f"Nano-banana: {log['message']}")
        
        # Call nano-banana edit
        #prompt = f"Generate a similar scene with the same aesthetic and lighting as the reference images, but place the person from the AI host image into this setting. Maintain the atmosphere and visual style of the original scene."
        prompt = f"This image is a screenshot from a podcast studio where the host is talking. Imagine a new angle of the same studio with the man in brown sweater instead of the current person. The scene should look and match the aesthetic of the existing image. Also there should be no one else apart from the man in the brown sweater in the final image. The man in the brown sweater is looking in the same direction as the original person from the image."
        
        print(f"\nGenerating 4 scenes with prompt: {prompt}")
        
        result = fal_client.subscribe(
            "fal-ai/nano-banana/edit",
            arguments={
                "prompt": prompt,
                "image_urls": all_image_urls,
                "num_images": 4
            },
            with_logs=True,
            on_queue_update=on_queue_update,
        )
        
        return result
        
    except Exception as e:
        print(f"Warning: Scene generation failed: {e}")
        return None

def save_generated_images(result, output_dir):
    """Save generated images from nano-banana result to output directory."""
    if not result or 'images' not in result:
        print("No generated images to save")
        return []
    
    saved_files = []
    
    try:
        import requests
        
        for i, image_data in enumerate(result['images']):
            image_url = image_data['url']
            
            # Download the generated image
            response = requests.get(image_url)
            response.raise_for_status()
            
            # Save to output directory
            output_path = os.path.join(output_dir, f"generated_scene_{i+1}.jpg")
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            saved_files.append(output_path)
            print(f"  Saved generated scene: {output_path}")
    
    except Exception as e:
        print(f"Warning: Failed to save generated images: {e}")
    
    return saved_files

# ---------- Frame sampling function ----------
def sample_random_frames(
    video_path,
    out_dir="sampled_frames",
    num_frames=1
):
    """Sample 1 random frame from the middle portion of the video."""
    os.makedirs(out_dir, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    
    if total_frames == 0:
        raise RuntimeError("Could not determine video length")
    
    print(f"Video has {total_frames} frames at {fps:.2f} fps")
    
    # Sample from middle 60% of video (skip 20% from start and end)
    start_frame = int(total_frames * 0.2)
    end_frame = int(total_frames * 0.8)
    
    if end_frame - start_frame < num_frames:
        # If middle section is too short, expand the range
        start_frame = int(total_frames * 0.1)
        end_frame = int(total_frames * 0.9)
    
    # Always sample exactly 1 frame
    if end_frame - start_frame >= 1:
        frame_indices = [random.randint(start_frame, end_frame - 1)]
    else:
        # Fallback: use what we have
        frame_indices = [start_frame] if start_frame < end_frame else []
    
    saved_frames = []
    
    for i, frame_idx in enumerate(frame_indices):
        # Seek to specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Could not read frame {frame_idx}")
            continue
        
        # Save frame
        timestamp_s = frame_idx / fps
        out_path = os.path.join(out_dir, f"frame_{i+1}_{frame_idx:06d}.jpg")
        cv2.imwrite(out_path, frame)
        
        saved_frames.append({
            "file": out_path,
            "frame_idx": int(frame_idx),
            "timestamp_s": float(timestamp_s),
            "sample_number": i + 1
        })
    
    cap.release()
    return saved_frames

def main():
    parser = argparse.ArgumentParser(description='Sample 2 random frames from video and copy AI host image')
    parser.add_argument('input', help='Video file path or YouTube URL')
    parser.add_argument('ai_host_img', help='Path to AI host image file')
    parser.add_argument('--output-dir', '-o', default='sampled_frames', 
                        help='Output directory for sampled frames (default: sampled_frames)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable video caching for YouTube downloads')
    
    args = parser.parse_args()
    
    input_path = args.input
    ai_host_img_path = args.ai_host_img
    temp_video_path = None
    is_temporary = False
    
    # Validate AI host image exists
    if not os.path.exists(ai_host_img_path):
        print(f"Error: AI host image not found: {ai_host_img_path}")
        return 1
    
    
    try:
        # Check if input is a YouTube URL
        if input_path.startswith(('http://', 'https://')) and ('youtube.com' in input_path or 'youtu.be' in input_path):
            print(f"Downloading YouTube video: {input_path}")
            video_path, is_temporary = download_youtube_video(input_path, use_cache=not args.no_cache)
            if not is_temporary:
                print(f"Using cached video: {video_path}")
            else:
                print(f"Downloaded to: {video_path}")
                temp_video_path = video_path
        else:
            # Local file
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Video file not found: {input_path}")
            video_path = input_path
        
        print(f"Processing video: {video_path}")
        print(f"Output directory: {args.output_dir}")
        
        # Sample random frames
        results = sample_random_frames(
            video_path=video_path,
            out_dir=args.output_dir
        )
        
        print(f"\nSampled {len(results)} frame:")
        for result in results:
            print(f"  Frame {result['sample_number']:03d}: {result['file']} "
                  f"(frame {result['frame_idx']}, {result['timestamp_s']:.1f}s)")
        
        # Copy AI host image to output directory
        ai_host_output = os.path.join(args.output_dir, "ai_host.jpg")
        shutil.copy2(ai_host_img_path, ai_host_output)
        print(f"\nCopied AI host image: {ai_host_output}")
        
        # Generate new scenes with AI host using nano-banana edit
        print(f"\n{'='*50}")
        print("GENERATING SCENES WITH AI HOST")
        print(f"{'='*50}")
        
        frame_paths = [result['file'] for result in results]
        generation_result = generate_scenes_with_ai_host(frame_paths, ai_host_img_path, args.output_dir)
        
        generated_files = []
        if generation_result:
            print(f"\nScene generation completed!")
            if 'description' in generation_result:
                print(f"Description: {generation_result['description']}")
            
            # Save generated images
            print(f"\nSaving generated images...")
            generated_files = save_generated_images(generation_result, args.output_dir)
            
            if generated_files:
                print(f"\nGenerated {len(generated_files)} new scenes:")
                for i, file_path in enumerate(generated_files):
                    print(f"  Scene {i+1}: {file_path}")
            else:
                print("No images were generated or saved")
        else:
            print("Scene generation failed - continuing with original files only")
        
        total_files = len(results) + 1 + len(generated_files)
        print(f"\nTotal files in {args.output_dir}: {total_files} (1 frame + 1 AI host + {len(generated_files)} generated scenes)")
                  
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    finally:
        # Clean up temporary files only if they are actually temporary
        if temp_video_path and is_temporary and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except OSError:
                pass
    
    return 0

if __name__ == "__main__":
    exit(main())


# Example usage:
# python face_extract.py "https://www.youtube.com/watch?v=VIDEO_ID" ai_host.jpg
# python face_extract.py local_video.mp4 my_host_image.png --output-dir frames

