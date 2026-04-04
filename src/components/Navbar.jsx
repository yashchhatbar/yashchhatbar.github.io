import React, { useEffect, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';

const Navbar = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [activeSection, setActiveSection] = useState('#hero');

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

  const isHome = location.pathname === '/';

  // ✅ Scroll detection for active section
  useEffect(() => {
    const sections = document.querySelectorAll('section');

    const handleScroll = () => {
      let current = '#hero';

      sections.forEach((section) => {
        const sectionTop = section.offsetTop;
        if (window.scrollY >= sectionTop - 100) {
          current = `#${section.getAttribute('id')}`;
        }
      });

      setActiveSection(current);
    };

    window.addEventListener('scroll', handleScroll);

    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    if (window.lucide) {
      window.lucide.createIcons();
    }
    closeMenu();
  }, [location]);

  const handleNavClick = (e, to, hash) => {
    if (isHome && hash) {
      e.preventDefault();
      const element = document.querySelector(hash);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
        closeMenu();
      }
    } else if (hash) {
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
          <svg width="24" height="24" viewBox="0 0 32 32" fill="none">
            <path d="M16 18V30M16 18L4 6M16 18L28 6"
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
        </Link>

        <div className={`nav-links ${isMenuOpen ? 'active' : ''}`}>

          {/* HOME */}
          <a
            href="#hero"
            className={`nav-link ${activeSection === '#hero' ? 'active' : ''}`}
            onClick={(e) => handleNavClick(e, '/', '#hero')}
          >
            Home
          </a>

          {/* PROJECTS */}
          <a
            href="#home-projects"
            className={`nav-link ${activeSection === '#home-projects' ? 'active' : ''}`}
            onClick={(e) => handleNavClick(e, '/', '#home-projects')}
          >
            Projects
          </a>

          {/* CONTACT */}
          <a
            href="#home-contact"
            className={`nav-link ${activeSection === '#home-contact' ? 'active' : ''}`}
            onClick={(e) => handleNavClick(e, '/', '#home-contact')}
          >
            Contact
          </a>

        </div>

        <div className="nav-right">
          <a href="/resume.pdf" className="btn-cta" download="Yash_Chhatbar_Resume.pdf">
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
