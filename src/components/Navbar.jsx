import React, { useEffect, useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';

const Navbar = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const location = useLocation();

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
    document.body.classList.toggle('menu-open');
  };

  const closeMenu = () => {
    setIsMenuOpen(false);
    document.body.classList.remove('menu-open');
  };

  useEffect(() => {
    if (window.lucide) {
      window.lucide.createIcons();
    }
    closeMenu();
  }, [location]);

  return (
    <div className="navbar-wrapper">
      <nav className="navbar" id="navbar" aria-label="Primary">

        {/* LOGO */}
        <NavLink to="/" className="logo" onClick={closeMenu}>
          <svg width="24" height="24" viewBox="0 0 32 32" fill="none">
            <path
              d="M16 18V30M16 18L4 6M16 18L28 6"
              stroke="currentColor"
              strokeWidth="4"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <div className="logo-text">
            <span className="logo-name">YASH CHHATBAR</span>
            <span className="logo-sub">AI/ML ENGINEER</span>
          </div>
        </NavLink>

        {/* NAV LINKS */}
        <div className={`nav-links ${isMenuOpen ? 'active' : ''}`}>

          <NavLink
            to="/"
            end
            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
            onClick={closeMenu}
          >
            Home
          </NavLink>

          <NavLink
            to="/projects"
            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
            onClick={closeMenu}
          >
            Projects
          </NavLink>

          <NavLink
            to="/contact"
            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
            onClick={closeMenu}
          >
            Contact
          </NavLink>

        </div>

        {/* RIGHT SIDE */}
        <div className="nav-right">
          <a
            href="/resume.pdf"
            className="btn-cta"
            download="Yash_Chhatbar_Resume.pdf"
          >
            Download Resume
          </a>

          <button
            className={`hamburger ${isMenuOpen ? 'active' : ''}`}
            type="button"
            onClick={toggleMenu}
          >
            <i className={`fas ${isMenuOpen ? 'fa-times' : 'fa-bars'}`}></i>
          </button>
        </div>

      </nav>
    </div>
  );
};

export default Navbar;
