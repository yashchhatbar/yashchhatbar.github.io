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
  const isProjectDetail = location.pathname.startsWith("/projects/") && location.pathname !== "/projects";

  useEffect(() => {

    // Initialize main app scripts
    initApp();

    // Initialize cursor + animations
    const cleanupCursor = initCursor();
    const cleanupAnims = initAnimations();

    // Handle section scrolling from navbar/footer
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

      // Scroll to top for normal page navigation
      window.scrollTo(0, 0);

    }

    return () => {
      if (cleanupCursor) cleanupCursor();
      if (cleanupAnims) cleanupAnims();
    };

  }, [location]);

  return (

    <div className="app-container">

      {/* Custom Cursor */}
      <div className="cursor-dot"></div>
      <div className="cursor-outline"></div>

      {/* Hide Navbar on Project Detail pages */}
      {!isProjectDetail && <Navbar />}

      <main>

        <Routes>

          <Route path="/" element={<Home />} />

          <Route path="/projects" element={<Projects />} />

          <Route path="/projects/:slug" element={<ProjectDetail />} />

          <Route path="/contact" element={<Contact />} />

        </Routes>

      </main>

      {/* Hide Footer on Project Detail pages */}
      {!isProjectDetail && <Footer />}

    </div>

  );

}

export default App;