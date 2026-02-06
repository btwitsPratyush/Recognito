"use client";

import React from "react";

const VideoFeed: React.FC = () => {
  const API_BASE_URL = "http://localhost:5001";

  return (
    <div className="video-container bg-black">
      <img
        src={`${API_BASE_URL}/video_feed`}
        alt="Live video feed with face recognition"
        className="w-full h-auto object-contain block"
        style={{ minHeight: "600px", maxHeight: "80vh" }}
      />
    </div>
  );
};

export default VideoFeed;
