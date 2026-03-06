"use client";

import { useEffect, useState } from "react";

export default function ThemeToggle() {
  const [theme, setTheme] = useState<"light" | "dark">("dark");

  useEffect(() => {
    const savedTheme = localStorage.getItem("theme") as "light" | "dark" | null;
    if (savedTheme) {
      setTheme(savedTheme);
      document.documentElement.setAttribute("data-theme", savedTheme);
    }
  }, []);

  const toggleTheme = () => {
    const newTheme = theme === "dark" ? "light" : "dark";
    setTheme(newTheme);
    document.documentElement.setAttribute("data-theme", newTheme);
    localStorage.setItem("theme", newTheme);
  };

  return (
    <button className="theme-toggle" onClick={toggleTheme} title="Toggle theme">
      {theme === "dark" ? "🌙" : "☀️"}
      <style jsx>{`
        .theme-toggle {
          background: none;
          border: 1px solid var(--border);
          color: var(--foreground);
          padding: 4px 8px;
          border-radius: 6px;
          cursor: pointer;
          font-size: 16px;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: border-color 0.15s;
        }
        .theme-toggle:hover {
          border-color: var(--accent);
        }
      `}</style>
    </button>
  );
}
