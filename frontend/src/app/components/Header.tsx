"use client";

import React from "react";

const Header: React.FC = () => {
  return (
    <div className="text-center mb-12">
      <div className="inline-block relative">
        <h1
          className="text-7xl font-black tracking-tighter text-black select-none"
          style={{
            fontFamily: "'Geist', 'Inter', system-ui, sans-serif",
            letterSpacing: "-0.06em",
            lineHeight: 1,
          }}
        >
          <span
            style={{
              background:
                "linear-gradient(180deg, #000 0%, #000 50%, #444 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}
          >
            Rec
          </span>
          <span
            className="relative inline-block"
            style={{
              background:
                "linear-gradient(180deg, #000 0%, #000 50%, #444 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}
          >
            o
            <span
              className="absolute -top-1 -right-1 w-2.5 h-2.5 bg-black rounded-full"
              aria-hidden="true"
            />
          </span>
          <span
            style={{
              background:
                "linear-gradient(180deg, #000 0%, #000 50%, #444 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}
          >
            gnito
          </span>
        </h1>
        {/* Subtle underline accent */}
        <div
          className="mx-auto mt-2"
          style={{
            width: "60%",
            height: "3px",
            background:
              "linear-gradient(90deg, transparent, #000, transparent)",
            borderRadius: "2px",
          }}
        />
      </div>
      <h2 className="text-lg text-gray-500 font-medium mt-4 tracking-wide uppercase"
        style={{ letterSpacing: "0.15em", fontSize: "0.85rem" }}
      >
        Faces don&apos;t lie. Neither do I.
      </h2>
    </div>
  );
};

export default Header;
