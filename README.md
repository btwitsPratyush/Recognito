# Recognito

**Faces don't lie. Neither do I.**

Recognito is a real-time face recognition system powered by conversational AI. Point your camera at someone, and Recognito instantly identifies them and lets you have an AI-powered voice conversation with full context about who's in frame.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Features](#features)
3. [Tech Stack](#tech-stack)
4. [Prerequisites](#prerequisites)
5. [Project Structure](#project-structure)
6. [Setup](#setup)
7. [Running Recognito](#running-recogio)
8. [Adding New Profiles](#adding-new-profiles)
9. [Troubleshooting](#troubleshooting)

---

## How It Works <a name="how-it-works"></a>

```
Camera Feed -> Face Detection (OpenCV) -> Face Matching (dlib embeddings)
                                              |
                                              v
                                     Profile Lookup (JSON)
                                              |
                                              v
                              WebSocket -> Frontend Presence Bar
                                              |
                                              v
                                  ElevenLabs Voice AI (WebRTC)
```

1. **Backend** captures webcam frames, detects faces using HOG/CNN, and matches them against pre-computed embeddings.
2. **WebSocket** pushes recognized face data (name, profile, photo) to the frontend in real-time.
3. **Frontend** displays a live video feed, a presence bar showing who's visible, and an AI voice agent that knows who it's talking to.

---

## Features <a name="features"></a>

- **Real-time face recognition** with pre-computed embeddings for instant matching
- **Live presence bar** showing who's currently in frame with profile photos
- **AI voice agent** (ElevenLabs) that greets people by name and knows their background
- **Smoothing filter** that eliminates false detection blips (0.3s threshold)
- **WebRTC audio** for low-latency, high-quality voice conversations
- **`/whoisinframe` API** endpoint returning JSON of everyone currently visible
- **Scalable profiles** - add as many people as you want via simple folder + JSON structure

---

## Tech Stack <a name="tech-stack"></a>

| Layer      | Technology                                                  |
| ---------- | ----------------------------------------------------------- |
| Backend    | Python, FastAPI, OpenCV, dlib, face_recognition, NumPy      |
| Frontend   | Next.js 15, React 19, TypeScript, Tailwind CSS              |
| Voice AI   | ElevenLabs Conversational AI (WebRTC)                       |
| Real-time  | WebSockets (face updates), Server-Sent Events (video feed)  |
| Embeddings | Pickle-serialized NumPy arrays for fast vectorized matching |

---

## Prerequisites <a name="prerequisites"></a>

| Requirement | Version / Notes                          |
| ----------- | ---------------------------------------- |
| Python      | 3.9+                                     |
| Node.js     | 18+                                      |
| cmake       | Required for building dlib               |
| Web camera  | USB or built-in                          |
| ElevenLabs  | Free-tier agent ID works                 |

> **Tip:** Use a virtual environment: `python -m venv .venv && source .venv/bin/activate`

---

## Project Structure <a name="project-structure"></a>

```
recogio/
|-- backend_server.py          # FastAPI server - camera, face recognition, WebSocket
|-- create_embeddings.py       # One-time script to generate face embeddings
|-- face_embeddings.pkl        # Pre-computed face embeddings database
|-- requirements.txt           # Python dependencies
|
|-- profiles_images/           # Face photos (one folder per person)
|   +-- pratyushkumar/
|       +-- pratyushkumar_profile_picture.jpg
|
|-- profiles_data/             # Profile metadata (one JSON per person)
|   +-- pratyushkumar.json
|
+-- frontend/                  # Next.js 15 frontend
    +-- src/app/
        +-- components/
        |   |-- Header.tsx         # App header
        |   |-- VideoFeed.tsx      # Live camera stream
        |   +-- PresenceBar.tsx    # Face presence + AI voice agent
        |-- layout.tsx
        +-- page.tsx
    +-- .env.local                 # NEXT_PUBLIC_ELEVENLABS_AGENT_ID
    +-- package.json
```

---

## Setup <a name="setup"></a>

### 1. Install backend dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> On macOS, you may need `brew install cmake` before installing dlib.

### 2. Generate face embeddings

```bash
python create_embeddings.py    # scans profiles_images/* and builds face_embeddings.pkl
```

### 3. Install frontend dependencies

```bash
cd frontend && npm install
```

### 4. Configure ElevenLabs

Copy `.env.local.example` to `.env.local` and add your agent ID:

```env
NEXT_PUBLIC_ELEVENLABS_AGENT_ID=your_agent_id_here
```

Get a free agent ID at [elevenlabs.io/conversational-ai](https://elevenlabs.io/conversational-ai).

---

## Running Recognito <a name="running-recogio"></a>

| Terminal | Command                      | What it does                             |
| -------- | ---------------------------- | ---------------------------------------- |
| 1        | `python backend_server.py`   | Starts FastAPI backend on `localhost:5001`|
| 2        | `cd frontend && npm run dev` | Starts Next.js frontend on `localhost:3000`|

> **macOS users:** Run the backend from **Terminal.app** (not IDE terminal) to get the camera permission prompt.

Open [http://localhost:3000](http://localhost:3000), grant camera + microphone access, and you're live.

---

## Adding New Profiles <a name="adding-new-profiles"></a>

1. Create a folder in `profiles_images/` with the person's ID (e.g. `profiles_images/johndoe/`)
2. Add a clear face photo (JPG/PNG) inside that folder
3. Create a matching JSON in `profiles_data/johndoe.json`:

```json
{
  "profile_id": "johndoe",
  "name": "John Doe",
  "linkedin_url": "https://www.linkedin.com/in/johndoe",
  "about": "Software engineer passionate about AI.",
  "job_title": "Software Engineer",
  "company": "TechCorp",
  "experiences": [],
  "educations": [],
  "interests": ["AI", "Web Dev"],
  "accomplishments": []
}
```

4. Regenerate embeddings:

```bash
python create_embeddings.py
```

5. Restart the backend server.

---

## Troubleshooting <a name="troubleshooting"></a>

| Problem                        | Fix                                                                 |
| ------------------------------ | ------------------------------------------------------------------- |
| No camera / black screen       | Run backend from Terminal.app; check `cv2.VideoCapture(0)` index    |
| Embeddings missing             | Run `python create_embeddings.py`                                   |
| ElevenLabs not connecting      | Check `.env.local` has a valid agent ID                             |
| AI can't hear you              | Grant mic permission in Chrome; ensure no other app is using the mic|
| Port already in use            | `lsof -ti:5001 \| xargs kill -9` then restart                      |
| dlib won't install             | `brew install cmake` then retry `pip install dlib face-recognition` |

---

