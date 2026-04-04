import React from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";

const Footer = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const isHome = location.pathname === '/';

  const handleFooterLinkClick = (e, hash) => {
    if (isHome && hash.startsWith('#')) {
      e.preventDefault();
      const element = document.querySelector(hash);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
      }
    } else if (hash.startsWith('#')) {
      e.preventDefault();
      navigate('/');
      setTimeout(() => {
        const element = document.querySelector(hash);
        if (element) {
          element.scrollIntoView({ behavior: 'smooth' });
        }
      }, 100);
    }
  };

  return (
    <footer className="bcf-footer">
      <div className="bcf-container">

        {/* LEFT SIDE */}
        <div className="bcf-left">
          <div className="text-stage">
            <span id="layerIn">Hello</span>
          </div>

          <h1 className="bcf-title">
            YASH CHHATBAR <br />
            AI & ML ENGINEER
          </h1>

          <div className="bcf-copy">
            © {new Date().getFullYear()} Yash Chhatbar. All rights reserved.
          </div>
        </div>

        {/* RIGHT SIDE */}
        <div className="bcf-right">

          {/* QUICK LINKS */}
          <div className="bcf-col">
            <h4>Quick Links</h4>

            <ul>
              <li className="info-value">
                <a href="#hero" onClick={(e) => handleFooterLinkClick(e, '#hero')}>Home</a>
              </li>

              <li className="info-value">
                <a href="#about" onClick={(e) => handleFooterLinkClick(e, '#about')}>About</a>
              </li>

              <li className="info-value">
                <a href="#skills" onClick={(e) => handleFooterLinkClick(e, '#skills')}>Skills</a>
              </li>

              <li className="info-value">
                <Link to="/projects">All Projects</Link>
              </li>

              <li className="info-value">
                <a href="#experience" onClick={(e) => handleFooterLinkClick(e, '#experience')}>Experience</a>
              </li>

              <li className="info-value">
                <a href="#education" onClick={(e) => handleFooterLinkClick(e, '#education')}>Education</a>
              </li>
            </ul>
          </div>

          {/* SOCIAL LINKS */}
          <div className="bcf-col">
            <h4>Social Links</h4>

            <ul>

              <li className="info-value">
                <a href="mailto:yashchhatbar11@gmail.com">
                  Email Me
                </a>
              </li>

              <li className="info-value">
                <a
                  href="https://github.com/yashchhatbar"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  GitHub
                </a>
              </li>

              <li className="info-value">
                <a
                  href="https://www.linkedin.com/in/yashchhatbar/"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  LinkedIn
                </a>
              </li>

              <li className="info-value">
                <Link to="/contact">Contact</Link>
              </li>

              <li className="info-value">
                <a href="/resume.pdf" download="Yash_Chhatbar_Resume.pdf">
                  Download Resume
                </a>
              </li>

            </ul>

          </div>

        </div>
      </div>
    </footer>
  );
};

export default Footer;
