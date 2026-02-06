"use client";

import React, { useState, useEffect, useRef } from "react";
import { useConversation } from "@elevenlabs/react";

interface Face {
  name: string;
  slug: string;
  photo?: string;
  profile?: {
    profile_id?: string;
    name?: string;
    linkedin_url?: string;
    about?: string;
    job_title?: string;
    company?: string;
    title?: string;
    experiences?: Array<{
      title: string;
      company: string;
      location?: string;
      duration?: string;
      description?: string;
    }>;
    educations?: Array<{
      institution: string;
      degree: string;
      duration?: string;
      description?: string;
    }>;
    interests?: string[];
    accomplishments?: string[];
    recent_posts?: Array<{
      post_id: number;
      content: string;
      timestamp?: string;
      likes?: number;
      comments?: number;
      shares?: number;
      post_type?: string;
    }>;
    scraped_at?: string;
    profile_picture?: string;
    [key: string]: any; // For any additional fields
  };
  confidence: number;
}

interface FaceUpdateEvent {
  type: string;
  timestamp: number;
  faces: Face[];
}

const PresenceBar: React.FC = () => {
  const [currentFaces, setCurrentFaces] = useState<Face[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [hoveredFace, setHoveredFace] = useState<Face | null>(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [seenPeople, setSeenPeople] = useState<Set<string>>(new Set());
  const [callError, setCallError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(
    null
  );

  // Suppress benign livekit DataChannel console.error calls that trigger Next.js dev overlay
  useEffect(() => {
    const originalError = console.error;
    console.error = (...args: unknown[]) => {
      const msg = String(args[0] ?? "");
      if (
        msg.includes("Unknown DataChannel error") ||
        msg.includes("error_type") ||
        msg.includes("websocket closed") ||
        msg.includes("Server error:") ||
        msg.includes("Client tool with name")
      ) {
        // Downgrade to console.warn so it doesn't trigger Next.js error overlay
        console.warn("[suppressed]", ...args);
        return;
      }
      originalError.apply(console, args);
    };

    // Also suppress the window-level errors from the SDK
    const suppressSdkError = (event: ErrorEvent) => {
      if (
        event.message?.includes("error_type") ||
        event.message?.includes("Cannot read properties of undefined")
      ) {
        event.preventDefault();
        event.stopImmediatePropagation();
        return true;
      }
    };
    const suppressUnhandled = (event: PromiseRejectionEvent) => {
      const msg = String(event.reason);
      if (
        msg.includes("error_type") ||
        msg.includes("Cannot read properties of undefined")
      ) {
        event.preventDefault();
      }
    };
    window.addEventListener("error", suppressSdkError, true);
    window.addEventListener("unhandledrejection", suppressUnhandled, true);

    return () => {
      console.error = originalError;
      window.removeEventListener("error", suppressSdkError, true);
      window.removeEventListener("unhandledrejection", suppressUnhandled, true);
    };
  }, []);

  // ElevenLabs conversation hook with client tool for who_in_frame
  const conversation = useConversation({
    clientTools: {
      who_in_frame: async () => {
        try {
          const res = await fetch("http://localhost:5001/whoisinframe");
          const data = await res.json();
          console.log("who_in_frame tool called, result:", data);
          return JSON.stringify(data);
        } catch (err) {
          console.warn("who_in_frame fetch failed:", err);
          return JSON.stringify({ status: "error", message: "Backend not reachable" });
        }
      },
    },
    onConnect: () => {
      console.log("ElevenLabs conversation connected");
      setCallError(null);
    },
    onDisconnect: () => {
      console.log("ElevenLabs conversation disconnected");
    },
    onMessage: (message: any) => {
      console.log("ElevenLabs message:", message);
    },
    onError: (error: any) => {
      console.warn("ElevenLabs error:", error);
      setCallError("Conversation error: " + String(error));
    },
  });

  const connectWebSocket = () => {
    try {
      const ws = new WebSocket("ws://localhost:5001/ws/faces");
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        console.log("WebSocket connected");
      };

      ws.onmessage = (event) => {
        try {
          const data: FaceUpdateEvent = JSON.parse(event.data);
          if (data.type === "face_update") {
            // Check for new people
            const currentSlugs = new Set(currentFaces.map((face) => face.slug));
            const newPeople = data.faces.filter(
              (face) =>
                face.slug &&
                !seenPeople.has(face.slug) &&
                !currentSlugs.has(face.slug)
            );

            if (newPeople.length > 0 && conversation.status === "connected") {
              // Send comprehensive contextual update for new people
              console.log(
                "New people detected:",
                newPeople.map((p) => p.name).join(", ")
              );

              // Create detailed contextual update with ALL profile information
              const detailedUpdates = newPeople
                .map((person) => {
                  const profile = person.profile;
                  if (!profile)
                    return `${person.name} entered (no profile data available)`;

                  let update = `${person.name} just entered the frame.\n`;

                  // Basic info
                  if (profile.job_title)
                    update += `Current Role: ${profile.job_title}\n`;
                  if (profile.company)
                    update += `Company: ${profile.company}\n`;
                  if (profile.about)
                    update += `About: ${profile.about.substring(0, 200)}...\n`;

                  // Recent experience
                  if (profile.experiences && profile.experiences.length > 0) {
                    const recentExp = profile.experiences[0];
                    update += `Recent Experience: ${recentExp.title} at ${recentExp.company}`;
                    if (recentExp.duration)
                      update += ` (${recentExp.duration})`;
                    update += `\n`;
                  }

                  // Education
                  if (profile.educations && profile.educations.length > 0) {
                    const education = profile.educations[0];
                    update += `Education: ${education.degree} at ${education.institution}\n`;
                  }

                  // Recent activity
                  if (profile.recent_posts && profile.recent_posts.length > 0) {
                    const recentPost = profile.recent_posts[0];
                    update += `Recent Activity: ${recentPost.content.substring(
                      0,
                      100
                    )}... (${recentPost.timestamp})\n`;
                  }

                  return update;
                })
                .join("\n---\n");

              const contextualText = `NEW PEOPLE DETECTED:\n\n${detailedUpdates}`;

              // Use the official sendContextualUpdate method
              try {
                console.log("Sending contextual update:", contextualText);
                conversation.sendContextualUpdate(contextualText);
                console.log("Contextual update sent successfully!");
              } catch (error) {
                console.error("Failed to send contextual update:", error);
              }

              // Add new people to seen set
              newPeople.forEach((person) => {
                if (person.slug) seenPeople.add(person.slug);
              });
              setSeenPeople(new Set(seenPeople));
            }

            setCurrentFaces(data.faces);
          }
        } catch (error) {
          console.error("Error parsing WebSocket message:", error);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        console.log("WebSocket disconnected");

        // Attempt to reconnect after 3 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          console.log("Attempting to reconnect...");
          connectWebSocket();
        }, 3000);
      };

      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        setIsConnected(false);
      };
    } catch (error) {
      console.error("Failed to connect WebSocket:", error);
      setIsConnected(false);
    }
  };

  useEffect(() => {
    connectWebSocket();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const getInitials = (name: string): string => {
    return name
      .split(" ")
      .map((word) => word.charAt(0))
      .join("")
      .toUpperCase()
      .slice(0, 2);
  };

  const getAvatarBackground = (name: string): string => {
    // Generate consistent color based on name
    const colors = [
      "bg-blue-500",
      "bg-green-500",
      "bg-purple-500",
      "bg-pink-500",
      "bg-indigo-500",
      "bg-yellow-500",
      "bg-red-500",
      "bg-teal-500",
    ];

    const index = name
      .split("")
      .reduce((acc, char) => acc + char.charCodeAt(0), 0);
    return colors[index % colors.length];
  };

  const handleMouseEnter = (face: Face, event: React.MouseEvent) => {
    setHoveredFace(face);
    setMousePosition({ x: event.clientX, y: event.clientY });
  };

  const handleMouseMove = (event: React.MouseEvent) => {
    setMousePosition({ x: event.clientX, y: event.clientY });
  };

  const handleMouseLeave = () => {
    setHoveredFace(null);
  };

  const requestMicrophonePermission = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      // Release the mic stream immediately so ElevenLabs SDK can use it
      stream.getTracks().forEach(track => track.stop());
      return true;
    } catch (error) {
      console.error("Microphone permission denied:", error);
      return false;
    }
  };

  const startCall = async () => {
    try {
      setCallError(null);

      // Get agent ID from environment
      const agentId = process.env.NEXT_PUBLIC_ELEVENLABS_AGENT_ID;
      if (!agentId) {
        throw new Error(
          "ElevenLabs Agent ID not configured. Please set NEXT_PUBLIC_ELEVENLABS_AGENT_ID in your .env.local file"
        );
      }

      // Request microphone permission
      const hasPermission = await requestMicrophonePermission();
      if (!hasPermission) {
        setCallError(
          "Microphone permission is required for voice conversation"
        );
        return;
      }

      // Create comprehensive system prompt with ALL profile data
      const peopleContext = currentFaces.map((face) => face.profile || {});

      // Get first names for greeting
      const firstNamesList = currentFaces
        .map((face) => face.name?.split(" ")[0])
        .filter(Boolean);

      // Create proper greeting based on number of people
      let greeting;
      if (firstNamesList.length === 0) {
        greeting = "Hi! I'm Recognito, your intelligent networking assistant.";
      } else if (firstNamesList.length === 1) {
        greeting = `Hi ${firstNamesList[0]}! I'm Recognito, your intelligent networking assistant.`;
      } else if (firstNamesList.length === 2) {
        greeting = `Hi ${firstNamesList[0]} and ${firstNamesList[1]}! I'm Recognito, your intelligent networking assistant.`;
      } else {
        const lastPerson = firstNamesList.pop();
        greeting = `Hi ${firstNamesList.join(
          ", "
        )}, and ${lastPerson}! I'm Recognito, your intelligent networking assistant.`;
      }

      const systemPrompt = `You are Recognito, an intelligent networking assistant that recognizes people in real-time. You have access to a live video feed and can see who's currently in the room. 

TAKE INITIATIVE FROM THE START. MAKE A JOKE ABOUT PEOPLE IN THE FRAME
WHEN YOU ARE JUST STARTING A CONVERSATION MAKE A JOKE ABOUT THE PEOPLE IN THE FRAME.

MATCH THE MOST INTERSTING FACTS AND GIVE THEM TO USERS


IMPORTANT NOTES:
DON'T BE VERBOSE
BE CHILL AND UNHINGED, START WITH A JOKE ABOUT PEOPLE YOU SEE IN THE FRAME
DON'T BE TOO SERIOUS, BE FUNNY AND JOKEY
DON'T REPEAT YOURSELF. NEVER REPEAT YOURSELF

IF YOU ARE ASKED ABOUT THE PEOPLE IN THE FRAME, JUST CALL THE TOOL who_in_frame.
DON'T ASK CLARIFYING QUESTIONS, JUST CALL THE TOOL.


THE RECENT POST AND ACTIVITY ARE IMPORTANT FACTS TO USE. MENTION THEM IN CONVERSATION
TO USERS

FOR EXAMPLE "xyz has won a hackathon a couple of days ago" or "xyz went to SF last week, you were there when you worked for abc"

COMPLETE PROFILE DATA FOR PEOPLE CURRENTLY IN FRAME:
${peopleContext
          .map((profile, index) => {
            if (!profile || Object.keys(profile).length === 0)
              return `Person ${index + 1}: No profile data available`;

            let profileInfo = `\n=== ${profile.name || "Unknown"} ===\n`;

            // Basic info
            if (profile.about) profileInfo += `About: ${profile.about}\n`;
            if (profile.job_title)
              profileInfo += `Current Role: ${profile.job_title}\n`;
            if (profile.company) profileInfo += `Company: ${profile.company}\n`;
            if (profile.linkedin_url)
              profileInfo += `LinkedIn: ${profile.linkedin_url}\n`;

            // Experiences
            if (profile.experiences && profile.experiences.length > 0) {
              profileInfo += `\nWork Experience:\n`;
              profile.experiences.slice(0, 3).forEach((exp, i) => {
                profileInfo += `  ${i + 1}. ${exp.title} at ${exp.company}`;
                if (exp.duration) profileInfo += ` (${exp.duration})`;
                if (exp.location) profileInfo += ` - ${exp.location}`;
                if (exp.description)
                  profileInfo += `\n     ${exp.description.substring(0, 100)}...`;
                profileInfo += `\n`;
              });
            }

            // Education
            if (profile.educations && profile.educations.length > 0) {
              profileInfo += `\nEducation:\n`;
              profile.educations.slice(0, 2).forEach((edu, i) => {
                profileInfo += `  ${i + 1}. ${edu.degree} at ${edu.institution}`;
                if (edu.duration) profileInfo += ` (${edu.duration})`;
                if (edu.description) profileInfo += ` - ${edu.description}`;
                profileInfo += `\n`;
              });
            }

            // Recent posts/activity

            if (profile.recent_posts && profile.recent_posts.length > 0) {
              profileInfo += `\nRecent Activity:\n`;
              profile.recent_posts.slice(0, 2).forEach((post, i) => {
                profileInfo += `  ${i + 1}. ${post.content.substring(0, 150)}...`;
                if (post.timestamp) profileInfo += ` (${post.timestamp})`;
                profileInfo += `\n`;
              });
            }

            return profileInfo;
          })
          .join("\n")}

You are friendly, conversational

Keep your responses natural and conversational. You're like a well-informed friend at a networking event who knows everyone's background intimately.`;

      // Start the conversation using the ElevenLabs hook
      console.log("Starting ElevenLabs session with agentId:", agentId);

      try {
        const sessionId = await conversation.startSession({
          agentId,
          connectionType: "webrtc",
        });
        console.log("Session started:", sessionId);

        // Send face context as a contextual update after connection
        setTimeout(() => {
          if (conversation.status === "connected") {
            console.log("Sending contextual update with face data...");
            conversation.sendContextualUpdate(systemPrompt);
          }
        }, 2000);

      } catch (sessionError) {
        console.error("startSession error:", sessionError);
        throw sessionError;
      }

      // Mark all current people as seen
      currentFaces.forEach((face) => {
        if (face.slug) seenPeople.add(face.slug);
      });
      setSeenPeople(new Set(seenPeople));
    } catch (error) {
      console.error("Failed to start call:", error);
      setCallError(
        error instanceof Error ? error.message : "Failed to start call"
      );
    }
  };

  const endCall = async () => {
    try {
      await conversation.endSession();
    } catch (error) {
      console.error("Error ending call:", error);
    }
  };

  return (
    <div className="relative sticky top-6">
      <div className="w-full bg-white border border-gray-200 rounded-2xl shadow-xl p-6 backdrop-blur-sm bg-white/80 min-h-[500px]">
        {/* Call Recognito Button */}
        <div className="mb-6">
          <button
            onClick={conversation.status === "connected" ? endCall : startCall}
            disabled={!isConnected}
            className={`w-full flex items-center justify-center space-x-2 px-4 py-3 rounded-xl font-semibold text-white transition-all duration-200 ${conversation.status === "connected"
              ? "bg-red-600 hover:bg-red-700"
              : "bg-black hover:bg-gray-800"
              } disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              {conversation.status === "connected" ? (
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M16 8l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2M3 3l1.5 1.5M3 3v6m0 0l6-6m-6 6h6"
                />
              ) : (
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z"
                />
              )}
            </svg>
            <span>
              {conversation.status === "connected" ? "End Call" : "Start Conversation"}
            </span>
          </button>

          {/* Error display */}
          {callError && (
            <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-600">{callError}</p>
              {(callError.includes("Agent ID") ||
                callError.includes("configuration")) && (
                  <p className="text-xs text-red-500 mt-1">
                    Please set NEXT_PUBLIC_ELEVENLABS_AGENT_ID in your
                    frontend/.env.local file
                  </p>
                )}
            </div>
          )}

          {/* Call status indicator */}
          {conversation.status === "connected" && (
            <div className="mt-3 flex items-center justify-center space-x-2 text-green-600">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-sm font-medium">Call Active</span>
            </div>
          )}
        </div>

        <div className="flex items-center justify-between mb-6">
          <h3 className="text-2xl font-bold text-gray-900">Who's Here</h3>
          <div className="flex items-center space-x-3">
            <div
              className={`w-3 h-3 rounded-full ${isConnected ? "bg-green-500" : "bg-red-500"
                } animate-pulse`}
            ></div>
            <span className="text-base text-gray-600 font-medium">
              {isConnected ? "Live" : "Disconnected"}
            </span>
          </div>
        </div>

        <div
          className="flex flex-col space-y-4 overflow-y-auto pr-2 max-h-80"
          style={{ minHeight: "200px" }}
        >
          {currentFaces.length === 0 ? (
            <div className="text-center py-12 px-4">
              <div className="w-16 h-16 bg-gray-200 rounded-full mx-auto mb-4 flex items-center justify-center">
                <svg
                  className="w-8 h-8 text-gray-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                  />
                </svg>
              </div>
              <p className="text-gray-500 text-base font-medium">
                Nobody in frame yet...
              </p>
            </div>
          ) : (
            currentFaces.map((face, index) => (
              <div
                key={`${face.slug}-${index}`}
                className="flex items-center space-x-4 p-3 rounded-xl bg-gray-50 cursor-pointer transform transition-all duration-200 hover:scale-105 hover:bg-gray-100"
                onMouseEnter={(e) => handleMouseEnter(face, e)}
                onMouseMove={handleMouseMove}
                onMouseLeave={handleMouseLeave}
              >
                <div className="relative flex-shrink-0">
                  {face.photo ? (
                    <img
                      src={face.photo}
                      alt={face.name}
                      className="w-16 h-16 rounded-full object-cover border-3 border-white shadow-lg"
                      onError={(e) => {
                        // Fallback to initials if image fails to load
                        const target = e.target as HTMLImageElement;
                        target.style.display = "none";
                        if (target.nextSibling) {
                          (target.nextSibling as HTMLElement).style.display =
                            "flex";
                        }
                      }}
                    />
                  ) : null}
                  <div
                    className={`w-16 h-16 rounded-full ${getAvatarBackground(
                      face.name
                    )} 
                            flex items-center justify-center text-white font-bold text-lg shadow-lg border-3 border-white
                            ${face.photo ? "hidden" : "flex"}`}
                  >
                    {getInitials(face.name)}
                  </div>

                  {/* Confidence indicator */}
                  <div
                    className="absolute -bottom-1 -right-1 w-6 h-6 bg-green-500 rounded-full 
                               flex items-center justify-center text-white text-xs font-bold shadow-lg"
                  >
                    <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M5 13l4 4L19 7"></path></svg>
                  </div>
                </div>

                <div className="flex-1 min-w-0">
                  <p className="text-base font-semibold text-gray-900 truncate">
                    {face.name}
                  </p>
                  {face.profile?.title && (
                    <p className="text-sm text-gray-600 truncate">
                      {face.profile.title}
                    </p>
                  )}
                  {face.profile?.company && (
                    <p className="text-xs text-gray-500 truncate mt-1">
                      {face.profile.company}
                    </p>
                  )}
                </div>
              </div>
            ))
          )}
        </div>

        {/* Show count when multiple people */}
        {currentFaces.length > 0 && (
          <div className="mt-6 pt-4 border-t border-gray-200 text-center">
            <span className="text-sm text-gray-600 font-medium bg-gray-100 px-3 py-1 rounded-full">
              {currentFaces.length}{" "}
              {currentFaces.length === 1 ? "person" : "people"} recognized
            </span>
          </div>
        )}
      </div>

      {/* Hover Tooltip */}
      {hoveredFace && (
        <div
          className="fixed z-50 bg-white border border-gray-300 rounded-xl shadow-2xl p-4 max-w-sm"
          style={{
            left: mousePosition.x + 10,
            top: mousePosition.y - 100,
            pointerEvents: "none",
          }}
        >
          <div className="flex items-center space-x-3 mb-3">
            {hoveredFace.photo ? (
              <img
                src={hoveredFace.photo}
                alt={hoveredFace.name}
                className="w-12 h-12 rounded-full object-cover border-2 border-gray-200"
              />
            ) : (
              <div
                className={`w-12 h-12 rounded-full ${getAvatarBackground(
                  hoveredFace.name
                )} 
                              flex items-center justify-center text-white font-bold text-sm`}
              >
                {getInitials(hoveredFace.name)}
              </div>
            )}
            <div>
              <h4 className="font-bold text-gray-900 text-lg">
                {hoveredFace.name}
              </h4>
              {hoveredFace.profile?.title && (
                <p className="text-sm text-gray-600">
                  {hoveredFace.profile.title}
                </p>
              )}
            </div>
          </div>

          {hoveredFace.profile && (
            <div className="space-y-2 text-sm">
              {hoveredFace.profile.company && (
                <div>
                  <span className="font-medium text-gray-700">Company:</span>
                  <span className="ml-2 text-gray-600">
                    {hoveredFace.profile.company}
                  </span>
                </div>
              )}
              <div>
                <span className="font-medium text-gray-700">Confidence:</span>
                <span className="ml-2 text-green-600 font-semibold">
                  {Math.round(hoveredFace.confidence * 100)}%
                </span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default PresenceBar;
