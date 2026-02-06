from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import face_recognition
import pickle
import os
import traceback
import json
from typing import List, Dict, Any, Set
import asyncio
import io
import base64
import time

app = FastAPI(title="Face Recognition Video Stream API")

# Add CORS middleware to allow frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # NextJS dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
embeddings_data = None
video_capture = None
websocket_clients: Set[WebSocket] = set()
current_faces: List[Dict] = []
last_faces_update = 0

# Smoothing mechanism to prevent hallucinations
SMOOTHING_THRESHOLD = 0.3  # 0.3 seconds
face_detection_buffer: Dict[str, Dict] = {}  # Track when each face was first/last seen
stable_faces: List[Dict] = []  # The stable face list sent to frontend

def load_embeddings_database():
    """Load the precomputed face embeddings database"""
    embeddings_file = "face_embeddings.pkl"
    
    print(f"Looking for embeddings file: {embeddings_file}")
    
    if not os.path.exists(embeddings_file):
        print(f"Embeddings database not found: {embeddings_file}")
        return None
    
    try:
        with open(embeddings_file, 'rb') as f:
            embeddings_data = pickle.load(f)
        
        print(f"Loaded embeddings database with {len(embeddings_data['face_encodings'])} profiles")
        return embeddings_data
        
    except Exception as e:
        print(f"Error loading embeddings database: {e}")
        return None

async def broadcast_face_update(faces_data: List[Dict]):
    """Broadcast face detection updates to all connected WebSocket clients"""
    if not websocket_clients:
        return
    
    # Prepare the payload with face data
    payload = {
        "type": "face_update",
        "timestamp": time.time(),
        "faces": faces_data
    }
    
    # Send to all connected clients
    disconnected_clients = set()
    for client in websocket_clients:
        try:
            await client.send_text(json.dumps(payload))
        except:
            disconnected_clients.add(client)
    
    # Remove disconnected clients
    websocket_clients.difference_update(disconnected_clients)

def get_profile_slug_from_metadata(profile_metadata: Dict) -> str:
    """Extract profile slug from metadata"""
    # Try to get slug from various possible fields
    if 'profile_id' in profile_metadata:
        return profile_metadata['profile_id']
    elif 'profile_slug' in profile_metadata:
        return profile_metadata['profile_slug']
    elif 'slug' in profile_metadata:
        return profile_metadata['slug']
    elif 'name' in profile_metadata:
        # Convert name to slug format as fallback
        return profile_metadata['name'].lower().replace(' ', '-')
    else:
        return 'unknown'

def load_profile_photo_base64(slug: str) -> str:
    """Load profile photo as base64 string"""
    # Try to find the profile photo in profiles_images directory
    photo_dir = f"profiles_images/{slug}"
    
    if os.path.exists(photo_dir):
        # Look for image files in the directory
        for file in os.listdir(photo_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                photo_path = os.path.join(photo_dir, file)
                try:
                    with open(photo_path, 'rb') as f:
                        photo_data = base64.b64encode(f.read()).decode('utf-8')
                        return f"data:image/jpeg;base64,{photo_data}"
                except Exception as e:
                    print(f"Error loading photo for {slug}: {e}")
    
    return None

def faces_changed(old_faces: List[Dict], new_faces: List[Dict]) -> bool:
    """Check if the face list has significantly changed"""
    if len(old_faces) != len(new_faces):
        return True
    
    # Create sets of recognized names for comparison
    old_names = {face['name'] for face in old_faces if face['name'] != 'Unknown'}
    new_names = {face['name'] for face in new_faces if face['name'] != 'Unknown'}
    
    return old_names != new_names

def smooth_face_detections(detected_faces: List[Dict]) -> tuple[List[Dict], bool]:
    """
    Apply smoothing to face detections to prevent hallucinations.
    Only return face changes after they've been stable for SMOOTHING_THRESHOLD seconds.
    
    Returns: (stable_faces_list, should_update_frontend)
    """
    global face_detection_buffer, stable_faces
    
    current_time = time.time()
    detected_names = {face['name'] for face in detected_faces if face['name'] != 'Unknown'}
    
    # Update buffer with current detections
    for face in detected_faces:
        if face['name'] != 'Unknown':
            face_name = face['name']
            
            if face_name not in face_detection_buffer:
                # New face detected - start tracking
                face_detection_buffer[face_name] = {
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'face_data': face,
                    'is_stable': False
                }
                print(f"Started tracking {face_name} (needs {SMOOTHING_THRESHOLD}s to become stable)")
            else:
                # Update existing face
                face_detection_buffer[face_name]['last_seen'] = current_time
                face_detection_buffer[face_name]['face_data'] = face  # Update with latest data
    
    # Check for faces that are no longer detected
    faces_to_remove = []
    for face_name, face_info in face_detection_buffer.items():
        if face_name not in detected_names:
            # Face not detected in current frame
            time_since_last_seen = current_time - face_info['last_seen']
            if time_since_last_seen >= SMOOTHING_THRESHOLD:
                # Face has been missing long enough - remove it
                faces_to_remove.append(face_name)
                print(f"Removing {face_name} (absent for {time_since_last_seen:.1f}s)")
    
    # Remove faces that have been absent long enough
    for face_name in faces_to_remove:
        del face_detection_buffer[face_name]
    
    # Build stable faces list (faces that have been detected long enough)
    new_stable_faces = []
    for face_name, face_info in face_detection_buffer.items():
        time_since_first_seen = current_time - face_info['first_seen']
        
        if time_since_first_seen >= SMOOTHING_THRESHOLD:
            # Face has been stable long enough
            if not face_info['is_stable']:
                print(f"{face_name} is now stable (detected for {time_since_first_seen:.1f}s)")
            face_info['is_stable'] = True
            new_stable_faces.append(face_info['face_data'])
    
    # Check if stable faces have changed
    should_update = faces_changed(stable_faces, new_stable_faces)
    if should_update:
        stable_names = [f['name'] for f in new_stable_faces]
        print(f"Updating frontend with stable faces: {stable_names}")
    
    stable_faces = new_stable_faces
    
    return stable_faces, should_update

def detect_faces_smart_live(image):
    """Smart face detection - EXACT SAME METHOD as face_live.py for optimal performance"""
    try:
        # Try HOG first (faster) - SAME AS face_live.py
        encodings_hog = face_recognition.face_encodings(image, model="hog")
        if len(encodings_hog) > 0:
            # Get face locations for drawing boxes
            face_locations = face_recognition.face_locations(image, model="hog")
            return face_locations, encodings_hog
        
        # If HOG fails, try CNN (more accurate but slower) - SAME AS face_live.py
        encodings_cnn = face_recognition.face_encodings(image, model="cnn")
        if len(encodings_cnn) > 0:
            # Get face locations for drawing boxes
            face_locations = face_recognition.face_locations(image, model="cnn")
            return face_locations, encodings_cnn
        
        return [], []
    except Exception as e:
        print(f"Face detection error: {e}")
        return [], []

def detect_and_recognize_faces(frame, known_face_encodings, profile_metadata):
    """Detect and recognize faces in the frame - OPTIMIZED VERSION from face_live.py"""
    # Resize frame for faster processing (same as face_live.py)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # Convert BGR to RGB (same as face_live.py)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Use smart detection method from face_live.py
    face_locations, face_encodings = detect_faces_smart_live(rgb_small_frame)
    
    face_data = []
    
    for face_encoding, face_location in zip(face_encodings, face_locations):
        name = "Unknown"
        profile_info = None
        confidence = 0
        
        if len(known_face_encodings) > 0:
            # Lightning-fast comparison using precomputed embeddings (vectorized) - SAME AS face_live.py
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(distances)
            best_distance = distances[best_match_index]
            
            # Use slightly higher tolerance for live video (more forgiving) - SAME AS face_live.py
            if best_distance < 0.65:
                profile_info = profile_metadata[best_match_index]
                name = profile_info.get('name', 'Unknown')
                confidence = 1 - best_distance
            else:
                # Show possible match with uncertainty if reasonably close - SAME AS face_live.py
                if best_distance < 0.8:
                    profile_info = profile_metadata[best_match_index]
                    name = f"{profile_info.get('name', 'Unknown')} (?)"
                    confidence = 1 - best_distance
        
        # Scale back up face locations since the frame we detected on was scaled to 1/4 size
        top, right, bottom, left = face_location
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        # Get slug and photo for WebSocket payload
        slug = get_profile_slug_from_metadata(profile_info) if profile_info else None
        photo_base64 = load_profile_photo_base64(slug) if slug else None
        
        # Ensure ALL profile data is passed (complete JSON data)
        complete_profile = profile_info if profile_info else {}
        
        face_data.append({
            'name': name,
            'location': [top, right, bottom, left],
            'profile': complete_profile,  # ALL JSON data from embeddings
            'confidence': confidence,
            'distance': best_distance if len(known_face_encodings) > 0 else 1.0,
            'slug': slug,
            'photo': photo_base64
        })
    
    return face_data

def draw_face_data(frame, face_data):
    """Draw face recognition results on the frame - SAME STYLE AS face_live.py"""
    for face in face_data:
        top, right, bottom, left = face['location']
        name = face['name']
        confidence = face['confidence']
        profile = face.get('profile', {})
        
        # Choose colors based on recognition status - SAME AS face_live.py
        if "?" in name:
            color = (0, 165, 255)  # Orange for uncertain matches
        elif name != "Unknown":
            color = (0, 0, 0)    # Green for confirmed matches
        else:
            color = (0, 0, 255)    # Red for unknown faces
        
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Prepare label with name and job info - SAME AS face_live.py
        if name != "Unknown" and profile and profile.get("title"):
            label = f"{name}\n{profile.get('title', '')}"
            if profile.get("company"):
                label += f"\n{profile.get('company', '')}"
        else:
            label = name
        
        # Draw labels (handle multi-line text) - SAME AS face_live.py
        lines = label.split('\n')
        y_offset = bottom + 20
        
        for line in lines:
            if line.strip():  # Only draw non-empty lines
                # Calculate text size for background rectangle
                (text_width, text_height), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                
                # Draw background rectangle for text
                cv2.rectangle(frame, (left, y_offset - text_height - 5), 
                             (left + text_width, y_offset + 5), color, -1)
                
                # Draw text
                cv2.putText(frame, line, (left, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_offset += text_height + 10
    
    return frame

async def generate_frames():
    """Generate video frames with face recognition - OPTIMIZED like face_live.py"""
    global embeddings_data, video_capture, current_faces, last_faces_update
    
    if not video_capture or not video_capture.isOpened():
        raise HTTPException(status_code=500, detail="Camera not available")
    
    if embeddings_data is None:
        raise HTTPException(status_code=500, detail="Face embeddings not loaded")
    
    known_face_encodings = embeddings_data['face_encodings']
    profile_metadata = embeddings_data['profile_metadata']
    
    # Initialize variables for optimized processing - SAME AS face_live.py
    face_data = []
    process_this_frame = True
    frame_count = 0
    
    print(f"Starting optimized video stream with {len(known_face_encodings)} profiles")
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        frame_count += 1
        
        try:
            # Only process every other frame of video to save time - SAME AS face_live.py
            if process_this_frame:
                # Perform face recognition only on this frame
                face_data = detect_and_recognize_faces(frame, known_face_encodings, profile_metadata)
                
                # Check if faces have changed and emit WebSocket event
                stable_faces, should_update = smooth_face_detections(face_data)
                
                if should_update:
                    current_faces = stable_faces.copy()
                    last_faces_update = time.time()
                    
                    # Prepare faces for WebSocket (only recognized faces, exclude Unknown)
                    # Include ALL profile data for comprehensive AI context
                    websocket_faces = []
                    for face in current_faces:
                        if face['name'] != 'Unknown' and not face['name'].endswith(' (?)'):
                            websocket_faces.append({
                                'name': face['name'],
                                'slug': face['slug'],
                                'photo': face['photo'],
                                'profile': face['profile'],  # Complete JSON profile data
                                'confidence': face['confidence']
                            })
                    
                    # Broadcast to WebSocket clients
                    await broadcast_face_update(websocket_faces)
            
            # Always draw results (using last detected face data) - SAME AS face_live.py
            frame = draw_face_data(frame, face_data)
            
            # Add performance info to frame - SAME AS face_live.py
            if frame_count % 30 == 0:  # Update every 30 frames
                cv2.putText(frame, f"Profiles: {len(known_face_encodings)}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Frame: {frame_count}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Optimized Mode", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Encode frame as JPEG with higher quality for better face visibility
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Toggle frame processing - SAME AS face_live.py
            process_this_frame = not process_this_frame
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            # Continue with previous face_data on error
            frame = draw_face_data(frame, face_data)
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Reduced sleep for better responsiveness
        await asyncio.sleep(0.02)  # ~50 FPS max, but actual processing is ~25 FPS due to frame skipping

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    global embeddings_data, video_capture
    
    print("Starting Face Recognition API Server...")
    
    # Load embeddings database
    embeddings_data = load_embeddings_database()
    if embeddings_data is None:
        print("Warning: Embeddings database not found. Face recognition will not work.")
    
    # Initialize camera
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Warning: Could not open camera")
    else:
        print("Camera initialized successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global video_capture
    if video_capture:
        video_capture.release()
    print("Server shutdown complete")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Face Recognition API is running",
        "camera_available": video_capture is not None and video_capture.isOpened(),
        "embeddings_loaded": embeddings_data is not None
    }

@app.get("/video_feed")
async def video_feed():
    """Video streaming endpoint"""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/whoisinframe")
async def who_is_in_frame():
    """Fast endpoint that returns current face information for ElevenLabs agent"""
    global stable_faces
    
    if not stable_faces:
        return {
            "status": "success",
            "count": 0,
            "people": [],
            "message": "No one currently in frame"
        }
    
    # Build response with flattened information
    people_info = []
    for face_data in stable_faces:
        if face_data['name'] != 'Unknown' and not face_data['name'].endswith(' (?)'):
            person_info = {
                "name": face_data['name'],
                "slug": face_data.get('slug', ''),
                "confidence": round(face_data.get('confidence', 0) * 100, 1),
                "job_title": "",
                "company": "",
                "location": "",
                "background": "",
                "about": "",
                "linkedin_url": "",
                "recent_exp_title": "",
                "recent_exp_company": "",
                "recent_exp_duration": "",
                "recent_exp_description": "",
                "education_degree": "",
                "education_institution": "",
                "education_duration": ""
            }
            
            # Add profile information if available
            profile = face_data.get('profile')
            if profile:
                person_info.update({
                    "job_title": profile.get('job_title', ''),
                    "company": profile.get('company', ''),
                    "location": profile.get('location', ''),
                    "background": profile.get('background', ''),
                    "about": profile.get('about', ''),
                    "linkedin_url": profile.get('linkedin_url', '')
                })
                
                # Add recent experience as flattened fields
                if profile.get('experiences') and len(profile['experiences']) > 0:
                    recent_exp = profile['experiences'][0]
                    person_info.update({
                        "recent_exp_title": recent_exp.get('title', ''),
                        "recent_exp_company": recent_exp.get('company', ''),
                        "recent_exp_duration": recent_exp.get('duration', ''),
                        "recent_exp_description": recent_exp.get('description', '')[:200] + "..." if recent_exp.get('description', '') else ''
                    })
                
                # Add education as flattened fields
                if profile.get('educations') and len(profile['educations']) > 0:
                    education = profile['educations'][0]
                    person_info.update({
                        "education_degree": education.get('degree', ''),
                        "education_institution": education.get('institution', ''),
                        "education_duration": education.get('duration', '')
                    })
            
            people_info.append(person_info)
    
    return {
        "status": "success",
        "count": len(people_info),
        "people": people_info,
        "timestamp": time.time()
    }

@app.get("/profiles")
async def get_profiles():
    """Get information about loaded profiles"""
    if embeddings_data is None:
        raise HTTPException(status_code=404, detail="Embeddings database not loaded")
    
    profiles = []
    for profile in embeddings_data['profile_metadata']:
        profiles.append({
            'name': profile.get('name', 'Unknown'),
            'title': profile.get('title', ''),
            'company': profile.get('company', ''),
            'location': profile.get('location', '')
        })
    
    return {
        "total_profiles": len(profiles),
        "profiles": profiles
    }

@app.websocket("/ws/faces")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time face detection events"""
    await websocket.accept()
    websocket_clients.add(websocket)
    
    try:
        # Send current faces immediately upon connection
        if current_faces:
            websocket_faces = []
            for face in current_faces:
                if face['name'] != 'Unknown' and not face['name'].endswith(' (?)'):
                    websocket_faces.append({
                        'name': face['name'],
                        'slug': face['slug'],
                        'photo': face['photo'],
                        'profile': face['profile'],
                        'confidence': face['confidence']
                    })
            
            payload = {
                "type": "face_update",
                "timestamp": time.time(),
                "faces": websocket_faces
            }
            await websocket.send_text(json.dumps(payload))
        
        # Keep connection alive and handle messages
        while True:
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
                
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        websocket_clients.discard(websocket)

@app.get("/profile_photo/{slug}")
async def get_profile_photo(slug: str):
    """Get profile photo by profile slug"""
    photo_dir = f"profiles_images/{slug}"

    if os.path.exists(photo_dir):
        # Look for image files in the directory
        for file in os.listdir(photo_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                photo_path = os.path.join(photo_dir, file)
                if os.path.exists(photo_path):
                    return FileResponse(photo_path, media_type="image/jpeg")
    
    # Return 404 if no photo found
    raise HTTPException(status_code=404, detail="Profile photo not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001) 