import os
import cv2
import numpy as np
import tempfile
import hashlib
import random
import shutil
import yt_dlp
import fal_client
import time
import base64
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up fal.ai API key from .env file
if 'FAL_API_KEY' in os.environ and 'FAL_KEY' not in os.environ:
    os.environ['FAL_KEY'] = os.environ['FAL_API_KEY']

class FaceExtractWebapp:
    def __init__(self):
        self.step_counter = 0
        self.all_videos = []  # Store videos for lipsync step
    
    def emit_step(self, title, status="active", message="", images=None, data=None, is_parallel=False, parallel_id=None, parallel_index=None, requires_action=False, action_buttons=None):
        """Emit a step update"""
        if not is_parallel:
            self.step_counter += 1
        
        step_data = {
            "step": self.step_counter if not is_parallel else f"{self.step_counter}.{parallel_index}",
            "title": title,
            "status": status,  # "active", "completed", "error", "waiting_action"
            "message": message,
            "images": images or [],
            "data": data or {},
            "timestamp": time.time(),
            "is_parallel": is_parallel,
            "parallel_id": parallel_id,
            "parallel_index": parallel_index,
            "requires_action": requires_action,
            "action_buttons": action_buttons or []
        }
        return step_data
    
    def get_cache_path(self, url):
        """Generate a cache file path based on the URL hash."""
        cache_dir = os.path.expanduser("~/.cache/face_extract")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create hash of the URL for unique filename
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return os.path.join(cache_dir, f"video_{url_hash}.mp4")

    def download_youtube_video(self, url, use_cache=True):
        """Download a YouTube video and return the path to the downloaded file."""
        if use_cache:
            cache_path = self.get_cache_path(url)
            if os.path.exists(cache_path):
                return cache_path, False  # False means not temporary
        
        try:
            if use_cache:
                # Download directly to cache path
                output_path = self.get_cache_path(url)
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

    def sample_random_frames(self, video_path, out_dir="sampled_frames"):
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
        
        # Sample from middle 60% of video (skip 20% from start and end)
        start_frame = int(total_frames * 0.2)
        end_frame = int(total_frames * 0.8)
        
        if end_frame - start_frame < 1:
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

    def generate_scenes_with_ai_host(self, frame_paths, ai_host_path, output_dir):
        """Generate new scenes using nano-banana edit with AI host in frame settings."""
        try:
            # Upload frame image (only 1 frame now)
            frame_urls = []
            for i, frame_path in enumerate(frame_paths):
                uploaded_frame = fal_client.upload_file(frame_path)
                frame_url = uploaded_frame if isinstance(uploaded_frame, str) else uploaded_frame.url
                frame_urls.append(frame_url)
            
            # Upload AI host image
            uploaded_host = fal_client.upload_file(ai_host_path)
            host_url = uploaded_host if isinstance(uploaded_host, str) else uploaded_host.url
            
            # Create all image URLs list (frames + host)
            all_image_urls = frame_urls + [host_url]
            
            # Call nano-banana edit
            prompt = f"This image is a screenshot from a podcast studio where the host is talking. Imagine a new angle of the same studio with the man in the white tshirt and green jacket of the current person. The scene should look and match the aesthetic of the existing image. Also there should be no one else apart from the man in the white shirt and green jacket in the final image. The man in the white shirt and green jacket is looking in the same direction as the original person from the image."
            
            result = fal_client.subscribe(
                "fal-ai/nano-banana/edit",
                arguments={
                    "prompt": prompt,
                    "image_urls": all_image_urls,
                    "num_images": 4
                },
                with_logs=False,  # Disable logs to avoid interference
            )
            
            return result, frame_urls, host_url
            
        except Exception as e:
            raise RuntimeError(f"Scene generation failed: {str(e)}")

    def save_generated_images(self, result, output_dir):
        """Save generated images from nano-banana result to output directory."""
        if not result or 'images' not in result:
            return []
        
        saved_files = []
        
        try:
            import requests
            
            # Debug: Print the actual response structure
            print("DEBUG: nano-banana response structure:")
            print(f"  Keys in result: {list(result.keys())}")
            if 'images' in result:
                print(f"  Number of images: {len(result['images'])}")
                if result['images']:
                    print(f"  Keys in first image: {list(result['images'][0].keys())}")
                    print(f"  First image data: {result['images'][0]}")
            
            for i, image_data in enumerate(result['images']):
                image_url = image_data['url']
                
                # Check if seed is actually present
                seed = image_data.get('seed', None)
                if seed is None:
                    # nano-banana doesn't provide seed, so we'll use index or generate our own identifier
                    seed = f"gen_{i+1}"
                    print(f"  No seed found for image {i+1}, using: {seed}")
                else:
                    print(f"  Found seed for image {i+1}: {seed}")
                
                # Download the generated image
                response = requests.get(image_url)
                response.raise_for_status()
                
                # Save to output directory
                output_path = os.path.join(output_dir, f"generated_scene_{i+1}_seed_{seed}.jpg")
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                saved_files.append({
                    "path": output_path,
                    "seed": seed,
                    "url": image_url,
                    "has_real_seed": seed is not None and not seed.startswith('gen_')
                })
        
        except Exception as e:
            raise RuntimeError(f"Failed to save generated images: {str(e)}")
        
        return saved_files

    def analyze_generations_with_openai(self, original_frame_path, ai_host_path, generated_files):
        """Analyze generated images using OpenAI Vision to select the best one."""
        try:
            # Initialize OpenAI client
            client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
            
            def encode_image(image_path):
                """Encode image to base64 for OpenAI API"""
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
            
            # Prepare images
            original_b64 = encode_image(original_frame_path)
            host_b64 = encode_image(ai_host_path)
            
            generated_images = []
            for i, file_info in enumerate(generated_files):
                if isinstance(file_info, dict):
                    path = file_info['path']
                else:
                    path = file_info
                generated_images.append({
                    'index': i,
                    'base64': encode_image(path),
                    'path': path
                })
            
            # Construct the analysis prompt
            prompt = """You are an expert in video production and AI-generated imagery. I need you to analyze these images and select the best generated scene.

Context:
- Image 1: Original podcast frame showing a person in the studio
- Image 2: AI host image (the person who should replace the original)
- Images 3-6: Four AI-generated scenes where the AI host has been placed into the podcast studio

Evaluation Criteria:
1. **Visual Integration**: How naturally is the AI host integrated into the scene?
2. **Lighting Consistency**: Does the lighting on the AI host match the original studio lighting?
3. **Scale & Perspective**: Is the AI host properly sized and positioned?
4. **Realism**: Does it look like a real person sitting in the studio vs. artificially inserted?
5. **Studio Coherence**: Is the podcast studio background/setting maintained properly?
6. **Professional Quality**: Which would look best in an actual broadcast?

Please analyze each generated image (Images 3-6) and provide:
1. A ranking from best (1) to worst (4)
2. Detailed analysis of each image's strengths and weaknesses
3. Your final recommendation with reasoning

Respond in JSON format:
{
  "best_image_index": 0,  // 0-3 (which of the 4 generations is best)
  "ranking": [0, 1, 2, 3],  // indices ordered from best to worst
  "analysis": {
    "image_0": "Detailed analysis of first generation...",
    "image_1": "Detailed analysis of second generation...",
    "image_2": "Detailed analysis of third generation...",
    "image_3": "Detailed analysis of fourth generation..."
  },
  "reasoning": "Overall reasoning for why the best image was selected...",
  "confidence": 0.85  // 0-1 confidence score
}"""

            # Prepare messages with images
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{original_b64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{host_b64}"}}
                    ]
                }
            ]
            
            # Add generated images
            for img in generated_images:
                messages[0]["content"].append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{img['base64']}"}
                })
            
            # Call OpenAI Vision API
            response = client.chat.completions.create(
                model="gpt-4o",  # Latest multimodal model
                messages=messages,
                max_tokens=1500,
                temperature=0.3
            )
            
            # Parse response
            analysis_text = response.choices[0].message.content
            
            # Try to extract JSON from response
            try:
                # Look for JSON in the response
                json_start = analysis_text.find('{')
                json_end = analysis_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    analysis_json = json.loads(analysis_text[json_start:json_end])
                else:
                    # Fallback if no JSON found
                    analysis_json = {
                        "best_image_index": 0,
                        "ranking": [0, 1, 2, 3],
                        "analysis": {"summary": analysis_text},
                        "reasoning": "Analysis completed but structured response unavailable",
                        "confidence": 0.7
                    }
            except json.JSONDecodeError:
                # Fallback parsing
                analysis_json = {
                    "best_image_index": 0,
                    "ranking": [0, 1, 2, 3],
                    "analysis": {"summary": analysis_text},
                    "reasoning": "Analysis completed but JSON parsing failed",
                    "confidence": 0.6
                }
            
            return analysis_json
            
        except Exception as e:
            print(f"OpenAI analysis failed: {e}")
            # Return fallback analysis
            return {
                "best_image_index": 0,
                "ranking": [0, 1, 2, 3],
                "analysis": {
                    "image_0": "Analysis unavailable - OpenAI API error",
                    "image_1": "Analysis unavailable - OpenAI API error", 
                    "image_2": "Analysis unavailable - OpenAI API error",
                    "image_3": "Analysis unavailable - OpenAI API error"
                },
                "reasoning": f"OpenAI analysis failed: {str(e)}",
                "confidence": 0.0,
                "error": True
            }

    def animate_best_image_with_veo2(self, image_path, prompt="The person at the table is talking and interacting intently.", duration=5):
        """Animate the best selected image using VEO2 image-to-video model.
        
        Args:
            image_path (str): Path to the image to animate
            prompt (str): Text prompt describing the animation
            duration (int): Video duration in seconds (default: 5)
            
        Returns:
            dict: VEO2 result containing video URL and metadata
        """
        try:
            # Upload the best image to fal.ai
            uploaded_image = fal_client.upload_file(image_path)
            image_url = uploaded_image if isinstance(uploaded_image, str) else uploaded_image.url
            
            # Define queue update handler for progress logging
            def on_queue_update(update):
                if isinstance(update, fal_client.InProgress):
                    for log in update.logs:
                        print(f"VEO2 Progress: {log['message']}")
            
            # Call VEO2 image-to-video model
            result = fal_client.subscribe(
                "fal-ai/veo2/image-to-video",
                arguments={
                    "prompt": prompt,
                    "image_url": image_url,
                    "duration": duration
                },
                with_logs=True,
                on_queue_update=on_queue_update,
            )
            
            return {
                "success": True,
                "video_result": result,
                "input_image_url": image_url,
                "prompt": prompt,
                "duration": duration
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "input_image_path": image_path,
                "prompt": prompt,
                "duration": duration
            }


    def apply_parallel_lipsync_to_talking_video(self, talking_video_url, output_dir):
        """Apply lipsync to the talking animation video using Kling API with two different texts in parallel.
        
        Args:
            talking_video_url (str): URL of the talking animation video from VEO2
            output_dir (str): Directory to save results
            
        Returns:
            dict: Results containing both lipsynced video URLs and metadata
        """
        import requests
        import tempfile
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # First, download the VEO2 video and upload it to fal.ai
        try:
            print(f"Downloading VEO2 video from: {talking_video_url}")
            response = requests.get(talking_video_url)
            response.raise_for_status()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                tmp_file.write(response.content)
                temp_video_path = tmp_file.name
            
            print(f"Uploading video to fal.ai...")
            # Upload to fal.ai to get a fal.media URL
            uploaded_video = fal_client.upload_file(temp_video_path)
            fal_video_url = uploaded_video if isinstance(uploaded_video, str) else uploaded_video.url
            print(f"Video uploaded to fal.ai: {fal_video_url}")
            
            # Clean up temp file
            os.unlink(temp_video_path)
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to upload video to fal.ai: {str(e)}",
                "original_video_url": talking_video_url,
                "lipsync_results": []
            }
        
        # The two texts to be spoken
        texts = [
            {
                "id": 1,
                "text": "Simply put, the idea shifts from monthly payments to giving everyone ownership shares and real decision-making power.",
                "filename": "talking_lipsync_1.mp4"
            },
            {
                "id": 2,
                "text": "In short, distributing AI-created value can be designed as a global system, not limited by where technology originated.",
                "filename": "talking_lipsync_2.mp4"
            }
        ]
        
        def process_single_lipsync(text_info):
            """Process a single lipsync request"""
            try:
                # Define queue update handler for progress logging
                def on_queue_update(update):
                    if isinstance(update, fal_client.InProgress):
                        for log in update.logs:
                            print(f"Lipsync {text_info['id']} Progress: {log['message']}")
                
                # Call Kling lipsync API with the fal.media URL
                result = fal_client.subscribe(
                    "fal-ai/kling-video/lipsync/text-to-video",
                    arguments={
                        "video_url": fal_video_url,  # Use the uploaded fal.media URL
                        "text": text_info['text'],
                        "voice_id": "oversea_male1"  # Using oversea_male1 voice
                    },
                    with_logs=True,
                    on_queue_update=on_queue_update
                )
                
                # Download the lipsynced video if successful
                if result and 'video' in result:
                    video_url = result['video']['url'] if isinstance(result['video'], dict) else result['video']
                    
                    # Download and save the lipsynced video
                    response = requests.get(video_url)
                    response.raise_for_status()
                    
                    output_path = os.path.join(output_dir, text_info['filename'])
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    
                    return {
                        "success": True,
                        "id": text_info['id'],
                        "video_url": video_url,
                        "local_path": output_path,
                        "text": text_info['text'],
                        "voice_id": "oversea_male1"
                    }
                else:
                    return {
                        "success": False,
                        "id": text_info['id'],
                        "error": "No video URL in Kling response",
                        "text": text_info['text']
                    }
                    
            except Exception as e:
                return {
                    "success": False,
                    "id": text_info['id'],
                    "error": str(e),
                    "text": text_info['text']
                }
        
        # Process both lipsyncs in parallel
        results = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both lipsync tasks
            future_to_text = {
                executor.submit(process_single_lipsync, text_info): text_info
                for text_info in texts
            }
            
            # Collect results
            for future in as_completed(future_to_text):
                text_info = future_to_text[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        "success": False,
                        "id": text_info['id'],
                        "error": str(e),
                        "text": text_info['text']
                    })
        
        # Sort results by ID to maintain order
        results.sort(key=lambda x: x['id'])
        
        # Return combined results
        return {
            "success": all(r['success'] for r in results),
            "original_video_url": talking_video_url,
            "lipsync_results": results
        }

    def create_parallel_animations(self, best_image_path, unused_param, output_dir):
        """Create four parallel VEO2 animations with different actions.
        
        Args:
            best_image_path (str): Path to the best generated image
            unused_param: Kept for compatibility (ignored)
            output_dir (str): Directory to save results
            
        Returns:
            dict: Results from all parallel animations
        """
        
        # Define the four animation configurations - all use the same best image
        animations = [
            {
                "name": "Talking",
                "source_image": best_image_path,
                "prompt": "The person at the table is talking and interacting intently.",
                "duration": 5,
                "parallel_index": 0
            },
            {
                "name": "Banana",
                "source_image": best_image_path,
                "prompt": "The person peels a banana and takes a bite of it.",
                "duration": 5,
                "parallel_index": 1
            },
            {
                "name": "Nodding",
                "source_image": best_image_path,
                "prompt": "The person is not speaking but nodding in agreement with a positive expression.",
                "duration": 5,
                "parallel_index": 2
            },
            {
                "name": "Disagreeing",
                "source_image": best_image_path,
                "prompt": "The person is not speaking and have their hands crossed, they are disagreeing by shaking their head slightly with a look of mild disapproval.",
                "duration": 5,
                "parallel_index": 3
            }
        ]
        
        parallel_results = {}
        parallel_id = f"parallel_{int(time.time())}"
        
        # Execute animations in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all animation tasks
            future_to_animation = {
                executor.submit(
                    self.animate_best_image_with_veo2,
                    anim['source_image'],
                    anim['prompt'],
                    anim['duration']
                ): anim
                for anim in animations
            }
            
            # Process completed animations
            for future in as_completed(future_to_animation):
                animation_info = future_to_animation[future]
                animation_name = animation_info['name']
                
                try:
                    result = future.result()
                    parallel_results[animation_name] = {
                        **result,
                        "animation_name": animation_name,
                        "parallel_index": animation_info['parallel_index'],
                        "parallel_id": parallel_id
                    }
                except Exception as e:
                    parallel_results[animation_name] = {
                        "success": False,
                        "error": str(e),
                        "animation_name": animation_name,
                        "parallel_index": animation_info['parallel_index'],
                        "parallel_id": parallel_id
                    }
        
        return {
            "parallel_id": parallel_id,
            "results": parallel_results,
            "total_animations": len(animations)
        }

    def process(self, video_url, ai_host_img_path, output_dir):
        """Main processing function that yields progress updates"""
        
        # Step 1: Validate inputs
        yield self.emit_step(
            "Validate Inputs", 
            "active", 
            "Validating AI host image and video URL"
        )
        
        if not os.path.exists(ai_host_img_path):
            yield self.emit_step(
                "Validate Inputs", 
                "error", 
                f"AI host image not found: {ai_host_img_path}"
            )
            return
        
        # Show AI host image
        yield self.emit_step(
            "Validate Inputs", 
            "completed", 
            "Input validation successful",
            images=[{"path": ai_host_img_path, "caption": "AI Host Image"}]
        )
        
        # Step 2: Download/Load Video
        yield self.emit_step(
            "Download Video", 
            "active", 
            f"Processing video from: {video_url}"
        )
        
        try:
            if video_url.startswith(('http://', 'https://')) and ('youtube.com' in video_url or 'youtu.be' in video_url):
                video_path, is_temporary = self.download_youtube_video(video_url, use_cache=True)
                if not is_temporary:
                    message = f"Using cached video: {os.path.basename(video_path)}"
                else:
                    message = f"Downloaded to: {os.path.basename(video_path)}"
            else:
                if not os.path.exists(video_url):
                    raise FileNotFoundError(f"Video file not found: {video_url}")
                video_path = video_url
                message = f"Using local video: {os.path.basename(video_path)}"
            
            yield self.emit_step(
                "Download Video", 
                "completed", 
                message
            )
            
        except Exception as e:
            yield self.emit_step(
                "Download Video", 
                "error", 
                str(e)
            )
            return
        
        # Step 3: Sample Frames
        yield self.emit_step(
            "Sample Frames", 
            "active", 
            "Sampling random frame from video"
        )
        
        try:
            results = self.sample_random_frames(video_path, output_dir)
            
            frame_images = []
            for result in results:
                frame_images.append({
                    "path": result['file'], 
                    "caption": f"Frame {result['sample_number']} (t={result['timestamp_s']:.1f}s)"
                })
            
            yield self.emit_step(
                "Sample Frames", 
                "completed", 
                f"Sampled {len(results)} frame from video",
                images=frame_images
            )
            
        except Exception as e:
            yield self.emit_step(
                "Sample Frames", 
                "error", 
                str(e)
            )
            return
        
        # Step 4: Copy AI Host Image
        yield self.emit_step(
            "Prepare Images", 
            "active", 
            "Copying AI host image to output directory"
        )
        
        # Preserve the original file extension
        _, ext = os.path.splitext(ai_host_img_path)
        ai_host_output = os.path.join(output_dir, f"ai_host{ext}")
        shutil.copy2(ai_host_img_path, ai_host_output)
        
        yield self.emit_step(
            "Prepare Images", 
            "completed", 
            "AI host image prepared",
            images=[{"path": ai_host_output, "caption": "AI Host (copied)"}]
        )
        
        # Step 5: Upload to fal.ai
        yield self.emit_step(
            "Upload to fal.ai", 
            "active", 
            "Uploading images to fal.ai for processing"
        )
        
        try:
            frame_paths = [result['file'] for result in results]
            generation_result, frame_urls, host_url = self.generate_scenes_with_ai_host(
                frame_paths, ai_host_img_path, output_dir
            )
            
            upload_data = {
                "frame_urls": frame_urls,
                "host_url": host_url
            }
            
            yield self.emit_step(
                "Upload to fal.ai", 
                "completed", 
                f"Successfully uploaded {len(frame_urls)} frame(s) and AI host image",
                data=upload_data
            )
            
        except Exception as e:
            yield self.emit_step(
                "Upload to fal.ai", 
                "error", 
                str(e)
            )
            return
        
        # Step 6: Generate Scenes
        yield self.emit_step(
            "Generate Scenes", 
            "active", 
            "Generating 4 new scenes with AI host..."
        )
        
        try:
            frame_paths = [result['file'] for result in results]
            generation_result, frame_urls_new, host_url_new = self.generate_scenes_with_ai_host(
                frame_paths,
                ai_host_img_path, 
                output_dir
            )
            
            if generation_result:
                # Save generated images
                generated_files = self.save_generated_images(generation_result, output_dir)
                
                generated_images = []
                for i, file_info in enumerate(generated_files):
                    if isinstance(file_info, dict):
                        # New structure with seed info
                        seed_info = f" (Seed: {file_info['seed']})" if file_info.get('has_real_seed') else f" (ID: {file_info['seed']})"
                        generated_images.append({
                            "path": file_info['path'],
                            "caption": f"Generated Scene {i+1}{seed_info}",
                            "seed": file_info['seed'],
                            "has_real_seed": file_info.get('has_real_seed', False)
                        })
                    else:
                        # Fallback for old structure
                        generated_images.append({
                            "path": file_info,
                            "caption": f"Generated Scene {i+1}",
                            "seed": "unknown",
                            "has_real_seed": False
                        })
                
                description = generation_result.get('description', 'Scene generation completed')
                
                yield self.emit_step(
                    "Generate Scenes", 
                    "completed", 
                    f"Generated {len(generated_files)} scenes. {description}",
                    images=generated_images
                )
            else:
                yield self.emit_step(
                    "Generate Scenes", 
                    "error", 
                    "Scene generation returned no results"
                )
                return
                
        except Exception as e:
            yield self.emit_step(
                "Generate Scenes", 
                "error", 
                str(e)
            )
            return
        
        # Step 7: AI Analysis
        yield self.emit_step(
            "Analyze Generations", 
            "active", 
            "Analyzing generated scenes with OpenAI Vision..."
        )
        
        try:
            analysis_result = self.analyze_generations_with_openai(
                frame_paths[0],  # Original frame
                ai_host_img_path,  # AI host image
                generated_files  # Generated images
            )
            
            # Debug: Print the analysis result
            print("DEBUG: OpenAI analysis result:")
            print(f"  Keys in result: {list(analysis_result.keys())}")
            print(f"  Best index: {analysis_result.get('best_image_index', 'not found')}")
            print(f"  Ranking: {analysis_result.get('ranking', 'not found')}")
            print(f"  Has error: {analysis_result.get('error', False)}")
            
            # Update generated images with analysis results
            analyzed_images = []
            
            # Validate best_image_index is within bounds
            best_idx = analysis_result.get('best_image_index', 0)
            if not isinstance(best_idx, int) or best_idx < 0 or best_idx >= len(generated_images):
                best_idx = 0  # Fallback to first image
                print(f"DEBUG: Invalid best_image_index {analysis_result.get('best_image_index')}, using 0")
            
            for i, img_info in enumerate(generated_images):
                is_best = (i == best_idx)
                
                # Safe ranking calculation with fallback
                try:
                    ranking_list = analysis_result.get('ranking', [0,1,2,3])
                    if i in ranking_list:
                        ranking_position = ranking_list.index(i) + 1
                    else:
                        ranking_position = i + 1  # Fallback to sequential numbering
                except (ValueError, AttributeError):
                    ranking_position = i + 1  # Fallback to sequential numbering
                
                analysis_text = ""
                if f'image_{i}' in analysis_result.get('analysis', {}):
                    analysis_text = analysis_result['analysis'][f'image_{i}']
                elif 'summary' in analysis_result.get('analysis', {}):
                    analysis_text = f"Position {ranking_position} in ranking"
                
                analyzed_images.append({
                    "path": img_info['path'],
                    "caption": img_info['caption'],
                    "seed": img_info.get('seed', 'unknown'),
                    "has_real_seed": img_info.get('has_real_seed', False),
                    "is_best": is_best,
                    "ranking": ranking_position,
                    "analysis": analysis_text,
                    "confidence": analysis_result.get('confidence', 0.0)
                })
            
            analysis_message = f"Analysis complete! Best image: Generation {analysis_result.get('best_image_index', 0) + 1}"
            if analysis_result.get('confidence'):
                analysis_message += f" (Confidence: {analysis_result['confidence']:.0%})"
            
            analysis_data = {
                "best_index": analysis_result.get('best_image_index', 0),
                "reasoning": analysis_result.get('reasoning', 'Analysis completed'),
                "confidence": analysis_result.get('confidence', 0.0),
                "has_error": analysis_result.get('error', False)
            }
            
            yield self.emit_step(
                "Analyze Generations", 
                "completed", 
                analysis_message,
                images=analyzed_images,
                data=analysis_data
            )
                
        except Exception as e:
            yield self.emit_step(
                "Analyze Generations", 
                "error", 
                f"Analysis failed: {str(e)}"
            )
            # Continue with original images if analysis fails
            analyzed_images = generated_images
        
        # Step 8: Create Parallel Animations
        yield self.emit_step(
            "Create Parallel Animations", 
            "active", 
            "Starting 4 parallel VEO2 animations..."
        )
        
        try:
            # Find the best image path
            best_image_path = None
            
            for img_info in analyzed_images:
                if img_info.get('is_best', False):
                    best_image_path = img_info['path']
                    break
            
            if not best_image_path:
                # Fallback to first generated image if no best image found
                for img_info in analyzed_images:
                    if 'generated_scene' in img_info['path']:
                        best_image_path = img_info['path']
                        break
            
            if best_image_path:
                # Start all parallel animations
                parallel_id = f"parallel_{int(time.time())}"
                
                # Emit parallel step indicators
                animations = [
                    {"name": "Talking", "message": "Creating talking animation (5s)..."},
                    {"name": "Banana", "message": "Creating banana eating animation (5s)..."},
                    {"name": "Nodding", "message": "Creating nodding animation (5s)..."},
                    {"name": "Disagreeing", "message": "Creating disagreeing animation (5s)..."}
                ]
                
                # Emit all parallel steps as active
                for i, anim in enumerate(animations):
                    yield self.emit_step(
                        f"Animate {anim['name']}",
                        "active",
                        anim['message'],
                        is_parallel=True,
                        parallel_id=parallel_id,
                        parallel_index=i
                    )
                
                # Execute parallel animations
                parallel_results = self.create_parallel_animations(best_image_path, best_image_path, output_dir)
                
                # Emit completion for each animation
                all_videos = []
                for animation_name, result in parallel_results['results'].items():
                    if result.get('success'):
                        video_result = result['video_result']
                        video_url = video_result.get('video', {}).get('url') if isinstance(video_result.get('video'), dict) else video_result.get('video')
                        
                        video_info = {
                            "video_url": video_url,
                            "duration": result['duration'],
                            "prompt": result['prompt'],
                            "animation_name": animation_name
                        }
                        all_videos.append(video_info)
                        
                        yield self.emit_step(
                            f"Animate {animation_name}",
                            "completed",
                            f"{animation_name} animation completed successfully!",
                            data=video_info,
                            is_parallel=True,
                            parallel_id=parallel_id,
                            parallel_index=result['parallel_index']
                        )
                    else:
                        yield self.emit_step(
                            f"Animate {animation_name}",
                            "error",
                            f"{animation_name} animation failed: {result.get('error', 'Unknown error')}",
                            is_parallel=True,
                            parallel_id=parallel_id,
                            parallel_index=result['parallel_index']
                        )
                
                # Store all_videos for lipsync step
                self.all_videos = all_videos
                
                # Emit overall completion
                yield self.emit_step(
                    "Create Parallel Animations",
                    "completed",
                    f"Successfully created {len(all_videos)}/4 parallel animations!",
                    data={
                        "all_videos": all_videos,
                        "total_animations": len(parallel_results['results']),
                        "successful_animations": len(all_videos)
                    }
                )
                
            else:
                yield self.emit_step(
                    "Create Parallel Animations",
                    "error",
                    "No suitable image found for animations"
                )
                
        except Exception as e:
            yield self.emit_step(
                "Create Parallel Animations",
                "error",
                f"Parallel animations failed: {str(e)}"
            )
        
        # Step 9: Apply Lipsync to Talking Video (2 parallel versions)
        yield self.emit_step(
            "Apply Lipsync",
            "active",
            "Creating 2 parallel lipsynced versions with different texts..."
        )
        
        try:
            # Find the talking video URL from the parallel animations results
            talking_video_url = None
            
            # Look for the talking video in self.all_videos
            for video_info in self.all_videos:
                if video_info.get('animation_name') == 'Talking':
                    talking_video_url = video_info.get('video_url')
                    break
            
            if talking_video_url:
                # Apply parallel lipsync to the talking video
                lipsync_result = self.apply_parallel_lipsync_to_talking_video(talking_video_url, output_dir)
                
                if lipsync_result['success']:
                    # Prepare data for both lipsynced videos
                    lipsync_data = {
                        "lipsync_videos": []
                    }
                    
                    for result in lipsync_result['lipsync_results']:
                        if result['success']:
                            lipsync_data["lipsync_videos"].append({
                                "id": result['id'],
                                "video_url": result['video_url'],
                                "text": result['text'],
                                "voice_id": result['voice_id']
                            })
                    
                    yield self.emit_step(
                        "Apply Lipsync",
                        "completed",
                        f"Successfully created {len(lipsync_data['lipsync_videos'])} lipsynced versions!",
                        data=lipsync_data
                    )
                else:
                    # Collect error messages
                    error_messages = []
                    for result in lipsync_result['lipsync_results']:
                        if not result['success']:
                            error_messages.append(f"Text {result['id']}: {result.get('error', 'Unknown error')}")
                    
                    yield self.emit_step(
                        "Apply Lipsync",
                        "error",
                        f"Lipsync failed: {'; '.join(error_messages)}"
                    )
            else:
                yield self.emit_step(
                    "Apply Lipsync",
                    "error",
                    "Could not find the talking animation video URL"
                )
                
        except Exception as e:
            yield self.emit_step(
                "Apply Lipsync",
                "error",
                f"Lipsync processing failed: {str(e)}"
            )
        
        # Step 10: Complete
        total_files = len(results) + 1 + len(generated_files)
        
        completion_message = f"Processing completed! Total files: {total_files} (1 frame + 1 AI host + {len(generated_files)} generated scenes)"
        
        yield self.emit_step(
            "Complete", 
            "completed", 
            completion_message,
            data={
                "total_files": total_files,
                "output_directory": output_dir
            }
        )
