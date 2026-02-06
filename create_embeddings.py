import face_recognition
import numpy as np
import os
import json
import pickle
from datetime import datetime

def load_profile_data(profile_id):
    """Load profile information from JSON file"""
    json_path = f"profiles_data/{profile_id}.json"
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Could not load profile data for {profile_id}: {e}")
        return {"name": profile_id, "job_title": "Unknown", "company": "Unknown"}

def detect_faces_smart(image, image_name):
    """Try HOG first (fast), then CNN (accurate) if no faces found"""
    print(f"   Detecting faces in {image_name}...")
    
    # Try HOG first (faster)
    encodings_hog = face_recognition.face_encodings(image, model="hog")
    if len(encodings_hog) > 0:
        print(f"      HOG found {len(encodings_hog)} face(s)")
        return encodings_hog
    
    # If HOG fails, try CNN (more accurate but slower)
    print(f"      HOG found 0 faces, trying CNN...")
    encodings_cnn = face_recognition.face_encodings(image, model="cnn")
    if len(encodings_cnn) > 0:
        print(f"      CNN found {len(encodings_cnn)} face(s)")
        return encodings_cnn
    
    print(f"      No faces found with either method")
    return []

def create_embeddings_database():
    """Create and save face embeddings database for all profiles"""
    print("Recognito - Embeddings Creator")
    print("=" * 50)
    print("This will process all profiles and create a face embeddings database")
    print("This may take a few minutes depending on the number of profiles...\n")
    
    # Get all profile directories
    profiles_dir = "profiles_images"
    if not os.path.exists(profiles_dir):
        print(f"Profiles directory not found: {profiles_dir}")
        return
    
    profile_dirs = [d for d in os.listdir(profiles_dir) 
                   if os.path.isdir(os.path.join(profiles_dir, d))]
    
    print(f"Found {len(profile_dirs)} profiles to process")
    
    # Storage for embeddings and metadata
    embeddings_data = {
        'face_encodings': [],
        'profile_metadata': [],
        'creation_date': datetime.now().isoformat(),
        'total_profiles': len(profile_dirs),
        'successful_profiles': 0,
        'failed_profiles': []
    }
    
    processed = 0
    successful = 0
    failed_profiles = []
    
    print("Starting face encoding process...\n")
    
    for profile_id in profile_dirs:
        processed += 1
        print(f"Processing {processed}/{len(profile_dirs)}: {profile_id}")
        
        try:
            profile_image_dir = os.path.join(profiles_dir, profile_id)
            
            # Find the profile image
            image_files = [f for f in os.listdir(profile_image_dir) 
                          if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            if not image_files:
                print(f"   No image found for {profile_id}")
                failed_profiles.append(f"{profile_id}: No image found")
                continue
            
            profile_image_path = os.path.join(profile_image_dir, image_files[0])
            
            # Load profile image
            profile_image = face_recognition.load_image_file(profile_image_path)
            
            # Detect faces in profile image
            profile_encodings = detect_faces_smart(profile_image, profile_id)
            
            if len(profile_encodings) == 0:
                print(f"   No face detected in {profile_id}")
                failed_profiles.append(f"{profile_id}: No face detected")
                continue
            
            # Use the first face encoding
            face_encoding = profile_encodings[0]
            
            # Load profile metadata
            profile_data = load_profile_data(profile_id)
            
            # Store encoding and metadata (ALL JSON data)
            embeddings_data['face_encodings'].append(face_encoding)
            
            # Include ALL profile data from JSON + image path
            complete_profile_data = profile_data.copy()
            complete_profile_data['image_path'] = profile_image_path
            
            embeddings_data['profile_metadata'].append(complete_profile_data)
            
            successful += 1
            print(f"   Successfully encoded: {profile_data.get('name', profile_id)}")
            
        except Exception as e:
            print(f"   Error processing {profile_id}: {e}")
            failed_profiles.append(f"{profile_id}: {str(e)}")
            continue
        
        # Progress indicator
        if processed % 10 == 0:
            print(f"   Progress: {processed}/{len(profile_dirs)} profiles processed, {successful} successful\n")
    
    # Update final statistics
    embeddings_data['successful_profiles'] = successful
    embeddings_data['failed_profiles'] = failed_profiles
    
    # Convert encodings to numpy array for better performance
    if embeddings_data['face_encodings']:
        embeddings_data['face_encodings'] = np.array(embeddings_data['face_encodings'])
    
    # Save the embeddings database
    output_file = "face_embeddings.pkl"
    print(f"\nSaving embeddings database to: {output_file}")
    
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(embeddings_data, f)
        print(f"Successfully saved embeddings database!")
    except Exception as e:
        print(f"Error saving embeddings database: {e}")
        return
    
    # Print final summary
    print(f"\n" + "=" * 60)
    print(f"EMBEDDINGS CREATION SUMMARY")
    print(f"=" * 60)
    print(f"   Total profiles found: {len(profile_dirs)}")
    print(f"   Successfully processed: {successful}")
    print(f"   Failed to process: {len(failed_profiles)}")
    print(f"   Success rate: {(successful/len(profile_dirs)*100):.1f}%")
    print(f"   Database file: {output_file}")
    print(f"   File size: {os.path.getsize(output_file)/1024:.1f} KB")
    print(f"   Creation date: {embeddings_data['creation_date']}")
    
    if failed_profiles:
        print(f"\nFAILED PROFILES:")
        for failure in failed_profiles[:10]:  # Show first 10 failures
            print(f"   - {failure}")
        if len(failed_profiles) > 10:
            print(f"   ... and {len(failed_profiles) - 10} more")
    
    print(f"\nEmbeddings database ready! You can now use 'face_fast.py' for instant face matching.")

if __name__ == "__main__":
    create_embeddings_database() 