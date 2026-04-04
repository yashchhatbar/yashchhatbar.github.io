import React, { useEffect } from "react";
import { Routes, Route, useLocation } from "react-router-dom";

import Navbar from "./components/Navbar";
import Footer from "./components/Footer";

import Home from "./pages/Home";
import Projects from "./pages/Projects";
import Contact from "./pages/Contact";
import ProjectDetail from "./pages/ProjectDetail";

import "./css/style.css";
import "./css/components.css";
import "./css/responsive.css";

import { initApp } from "./js/main.js";
import { initCursor } from "./js/interactions.js";
import { initAnimations } from "./js/animations.js";

function App() {

  const location = useLocation();

  // Detect if current page is a project detail page
  const isProjectDetail =
    location.pathname.startsWith("/projects/") &&
    location.pathname !== "/projects";

  useEffect(() => {

    // ✅ Initialize main scripts
    initApp();

    // ✅ Initialize cursor + animations
    const cleanupCursor = initCursor();
    const cleanupAnims = initAnimations();

    // ✅ Scroll behavior (FIXED)
    if (location.hash) {
      const element = document.querySelector(location.hash);

      if (element) {
        setTimeout(() => {
          element.scrollIntoView({
            behavior: "smooth",
            block: "start",
          });
        }, 100);
      }
    } else {
      // Always scroll to top on route change
      window.scrollTo({ top: 0, behavior: "smooth" });
    }

    return () => {
      if (cleanupCursor) cleanupCursor();
      if (cleanupAnims) cleanupAnims();
    };

  }, [location.pathname, location.hash]); // ✅ more precise dependency

  return (
    <div className="app-container">

      {/* Custom Cursor */}
      <div className="cursor-dot"></div>
      <div className="cursor-outline"></div>

      {/* Navbar (hidden on project detail page) */}
      {!isProjectDetail && <Navbar />}

      <main>
        <Routes>

          <Route path="/" element={<Home />} />

          <Route path="/projects" element={<Projects />} />

          <Route path="/projects/:slug" element={<ProjectDetail />} />

          <Route path="/contact" element={<Contact />} />

        </Routes>
      </main>

      {/* Footer (hidden on project detail page) */}
      {!isProjectDetail && <Footer />}

    </div>
  );
}

export default App;
