"use client";

import React from "react";
import Header from "./Header";
import VideoFeed from "./VideoFeed";
import PresenceBar from "./PresenceBar";

const VideoStream: React.FC = () => {
  return (
    <div className="min-h-screen bg-[#fafafa] py-8 flex flex-col">
      <div className="max-w-7xl mx-auto px-6 flex-1">
        <Header />

        <div className="flex gap-6 items-start">
          <div className="flex-1 min-w-0">
            <VideoFeed />
          </div>

          <div className="w-[340px] flex-shrink-0">
            <PresenceBar />
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="mt-16 pb-8 px-6">
        <div className="max-w-7xl mx-auto flex items-center justify-end gap-2">
          <div className="h-px flex-1 bg-gradient-to-r from-transparent to-gray-200" />
          <p className="text-sm text-gray-400 tracking-wide whitespace-nowrap">
            &copy; {new Date().getFullYear()} Recognito &middot; Developed by{" "}
            <span className="text-gray-900 font-semibold">Pratyush</span>
          </p>
        </div>
      </footer>
    </div>
  );
};

export default VideoStream;
