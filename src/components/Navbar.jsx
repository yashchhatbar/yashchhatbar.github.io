import React, { useEffect, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';

const Navbar = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
    if (!isMenuOpen) {
      document.body.classList.add('menu-open');
    } else {
      document.body.classList.remove('menu-open');
    }
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

  const isHome = location.pathname === '/';

  const handleNavClick = (e, to, hash) => {
    if (isHome && hash) {
      e.preventDefault();
      const element = document.querySelector(hash);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
        closeMenu();
      }
    } else if (hash) {
      // If we are on another page and want to go to a section on Home
      e.preventDefault();
      navigate('/');
      setTimeout(() => {
        const element = document.querySelector(hash);
        if (element) {
          element.scrollIntoView({ behavior: 'smooth' });
        }
      }, 100);
      closeMenu();
    } else {
      closeMenu();
    }
  };

  return (
    <div className="navbar-wrapper">
      <nav className="navbar" id="navbar" aria-label="Primary">
        <Link to="/" className="logo" onClick={closeMenu}>
          <svg width="24" height="24" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M16 18V30M16 18L4 6M16 18L28 6" stroke="currentColor" strokeWidth="4"
              strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <div className="logo-text">
            <span className="logo-name">YASH CHHATBAR</span>
            <span className="logo-sub">AI/ML ENGINEER</span>
          </div>
        </Link>

        <div className={`nav-links ${isMenuOpen ? 'active' : ''}`} id="primary-navigation">
          <a href="#hero" className={`nav-link ${isHome ? 'active' : ''}`} onClick={(e) => handleNavClick(e, '/', '#hero')}>Home</a>
          <a href="#home-projects" className="nav-link" onClick={(e) => handleNavClick(e, '/', '#home-projects')}>Projects</a>
          <a href="#home-contact" className="nav-link" onClick={(e) => handleNavClick(e, '/', '#home-contact')}>Contact</a>
        </div>

        <div className="nav-right">
          <a href="/resume.pdf" className="btn-cta" download="Yash_Chhatbar_Resume.pdf">Download Resume</a>
          <button
            className={`hamburger ${isMenuOpen ? 'active' : ''}`}
            type="button"
            aria-label="Open menu"
            aria-controls="primary-navigation"
            aria-expanded={isMenuOpen}
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