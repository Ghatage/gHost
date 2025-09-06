from flask import Flask, render_template, Response, send_from_directory, jsonify
import json
import time
import os
import threading
from face_extract_webapp import FaceExtractWebapp

app = Flask(__name__)

# Global variables to store progress
current_progress = []
processing_active = False

def process_video_task():
    """Background task to process video and send updates via SSE"""
    global current_progress, processing_active
    
    processing_active = True
    current_progress = []
    
    try:
        # Initialize the webapp processor
        processor = FaceExtractWebapp()
        
        # Hardcoded parameters for now - you can modify these
        video_url = "https://www.youtube.com/watch?v=-i9AGk3DJ90"
        ai_host_img = "./host.jpg"
        output_dir = "sampled_frames"
        
        # Process and yield updates
        for step_data in processor.process(video_url, ai_host_img, output_dir):
            current_progress.append(step_data)
            time.sleep(0.1)  # Small delay for smoother UI updates
            
    except Exception as e:
        error_step = {
            "step": len(current_progress) + 1,
            "title": "Error",
            "status": "error",
            "message": f"Error occurred: {str(e)}",
            "timestamp": time.time()
        }
        current_progress.append(error_step)
    
    finally:
        processing_active = False

@app.route('/')
def index():
    """Serve the main webapp page"""
    return render_template('index.html')

@app.route('/start')
def start_processing():
    """Start the video processing in background"""
    global processing_active
    
    if processing_active:
        return jsonify({"status": "already_running"})
    
    # Start processing in background thread
    thread = threading.Thread(target=process_video_task)
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "started"})

@app.route('/events')
def events():
    """Server-Sent Events endpoint for real-time updates"""
    def event_stream():
        last_sent = -1
        
        while True:
            # Send any new progress updates
            if len(current_progress) > last_sent + 1:
                for i in range(last_sent + 1, len(current_progress)):
                    step_data = current_progress[i]
                    yield f"data: {json.dumps(step_data)}\n\n"
                last_sent = len(current_progress) - 1
            
            # Check if processing is complete
            if not processing_active and len(current_progress) > 0:
                # Send completion signal
                completion_data = {
                    "type": "complete",
                    "message": "Processing completed"
                }
                yield f"data: {json.dumps(completion_data)}\n\n"
                break
                
            time.sleep(0.5)  # Poll every 500ms
    
    return Response(
        event_stream(), 
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*'
        }
    )

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve images from the output directory"""
    return send_from_directory('sampled_frames', filename)

@app.route('/status')
def get_status():
    """Get current processing status"""
    return jsonify({
        "active": processing_active,
        "steps_completed": len(current_progress),
        "current_progress": current_progress
    })

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('sampled_frames', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)