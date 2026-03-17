import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { projects } from "../data/projectsData";
import ProjectCard from "../components/ProjectCard";
import FAQ from "../components/FAQ";
import FAQSection from '../components/FAQ';

const Home = () => {
  useEffect(() => {
    // Re-initialize Lucide icons
    if (window.lucide) {
      window.lucide.createIcons();
    }

    // Dispatch event for external scripts to re-run their logic if necessary
    window.dispatchEvent(new Event('DOMContentLoaded'));
  }, []);

  return (
    <>
      <header id="hero" className="hero">
        <div className="hero-content">
          <span className="badge">Open to Work & Research</span>
          <h1 className="heroname">Yash Chhatbar</h1>
          <p className="hero-subline">AI & Machine Learning <span className="gradient-text">Engineer</span></p>
          <p className="tagline">Building intelligent systems to solve real-world problems.</p>
          <div className="cta-group">
            <a href="#home-projects" className="btn-primary" onClick={(e) => {
              e.preventDefault();
              document.getElementById('home-projects')?.scrollIntoView({ behavior: 'smooth' });
            }}>View Projects <i data-lucide="arrow-right"></i></a>
            <a href="#home-contact" className="btn-secondary" onClick={(e) => {
              e.preventDefault();
              document.getElementById('home-contact')?.scrollIntoView({ behavior: 'smooth' });
            }}>Contact Me</a>
          </div>
        </div>
        <div className="hero-visual">
          <div className="abstract-shape shape-1"></div>
          <div className="abstract-shape shape-2"></div>
          <div className="code-card glass">
            <div className="code-header">
              <span className="dot red"></span>
              <span className="dot yellow"></span>
              <span className="dot green"></span>
            </div>
            <pre><code>{`
def solve_problem(data):
    model = AI.load("future")
    insight = model.predict(data)
    return impact
            `}</code></pre>
          </div>
        </div>
      </header>

      <section id="about" className="section">
        <div className="container">
          <h2 className="section-title">About Me</h2>
          <div className="about-grid">
            <div className="about-text fade-in">
              <p>
                I’m an AI & Machine Learning Engineer focused on building practical, data-driven systems
                that solve real-world problems. My work spans machine learning, data analysis, and applied
                AI, with hands-on experience in transforming raw data into meaningful insights and
                deployable solutions.
              </p>
              <p>
                I value clean problem-solving, strong fundamentals, and continuous learning. Alongside
                industry experience, I’m actively preparing for advanced studies in Artificial Intelligence
                & Machine Learning, aiming to combine academic depth with real-world impact.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section id="research" className="section">
        <div className="container">
          <h2 className="section-title">Research Interests</h2>
          <div className="skills-grid">
            <article className="skill-category glass fade-in" data-role="ai">
              <h3><i data-lucide="brain"></i> Large Language Models</h3>
              <p className="project-desc">Prompt tuning, instruction fine-tuning, and RAG architectures for enterprise
                search.</p>
            </article>
            <article className="skill-category glass fade-in" data-role="ai backend">
              <h3><i data-lucide="bot"></i> AI Agents</h3>
              <p className="project-desc">Autonomous multi-agent systems, tool use capabilities, and reasoning loops.
              </p>
            </article>
            <article className="skill-category glass fade-in" data-role="ai data">
              <h3><i data-lucide="scan-eye"></i> Computer Vision</h3>
              <p className="project-desc">Face embeddings, zero-shot image classification, and high-performance video
                tracking.</p>
            </article>
          </div>
        </div>
      </section>

      <section id="skills" className="section">
        <div className="container">
          <h2 className="section-title">Technical Skills</h2>
          <div className="skills-grid">
            <div className="skill-category glass fade-in">
              <h3><i data-lucide="code-2"></i> Programming</h3>
              <div className="tags">
                <span>Python</span>
                <span>SQL</span>
                <span>JavaScript</span>
              </div>
            </div>
            <div className="skill-category glass fade-in">
              <h3><i data-lucide="brain-circuit"></i> Machine Learning & AI</h3>
              <div className="tags">
                <span>Machine Learning</span>
                <span>Deep Learning</span>
                <span>Computer Vision</span>
                <span>Model Evaluation</span>
                <span>Feature Engineering</span>
              </div>
            </div>
            <div className="skill-category glass fade-in">
              <h3><i data-lucide="layers"></i> Libraries & Frameworks</h3>
              <div className="tags">
                <span>TensorFlow</span>
                <span>PyTorch</span>
                <span>Scikit-learn</span>
                <span>OpenCV</span>
                <span>Pandas</span>
                <span>NumPy</span>
              </div>
            </div>
            <div className="skill-category glass fade-in">
              <h3><i data-lucide="bot"></i> LLM & AI Systems</h3>
              <div className="tags">
                <span>OpenAI (GPT-4, APIs)</span>
                <span>Prompt Engineering</span>
                <span>RAG Pipelines</span>
                <span>Embeddings</span>
                <span>LangChain</span>
                <span>Vector Databases (FAISS)</span>
              </div>
            </div>
            <div className="skill-category glass fade-in">
              <h3><i data-lucide="bar-chart-3"></i> Data & Visualization</h3>
              <div className="tags">
                <span>Power BI</span>
                <span>Excel</span>
                <span>Data Cleaning</span>
                <span>EDA</span>
              </div>
            </div>
            <div className="skill-category glass fade-in">
              <h3><i data-lucide="server"></i> Backend & Tools</h3>
              <div className="tags">
                <span>Flask</span>
                <span>FastAPI</span>
                <span>REST APIs</span>
                <span>Streamlit</span>
                <span>Git & GitHub</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="home-projects" className="section">

        <div className="container">

          <h2 className="section-title">Selected Work</h2>

          <div className="projects-grid">

            {projects.slice(0, 3).map(project => (

              <ProjectCard
                key={project.slug}
                project={project}
              />

            ))}

          </div>

          <div className="projects-cta">

            <Link to="/projects" className="btn-secondary">
              View All Projects
            </Link>

          </div>

        </div>

      </section>

      <section id="experience" className="section">
        <div className="container">
          <h2 className="section-title">Experience</h2>
          <div className="timeline">
            <div className="timeline-item">
              <div className="timeline-content">
                <span className="date">Jan 2026 – Present</span>
                <h3>AIML Intern</h3>
                <span className="institution">Intelivita Private Limited</span>
                <p>Worked on ML-driven solutions. Collaborated with cross-functional teams. Applied data-driven
                  decision making.</p>
              </div>
            </div>
            <div className="timeline-item">
              <div className="timeline-content">
                <span className="date">May 2025 – June 2025</span>
                <h3>Data Analyst & ML Intern</h3>
                <span className="institution">InfoLabz IT Services Pvt. Ltd.</span>
                <p>Data analysis and visualization. Model development and evaluation.</p>
              </div>
            </div>
            <div className="timeline-item">
              <div className="timeline-content">
                <span className="date">Feb 2023 – Sep 2023</span>
                <h3>Web Developer</h3>
                <span className="institution">Direct Leadz - Web Design</span>
                <p>Frontend & backend web development.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="education" className="section">
        <div className="container">
          <h2 className="section-title">Education</h2>
          <div className="timeline">
            <div className="timeline-item">
              <div className="timeline-content">
                <span className="date">Pursuing</span>
                <h3>Bachelor’s Degree in Information Technology</h3>
                <span className="institution">Gujarat Technological University</span>
                <p>Core subjects: ML, Data Structures, Databases. Strong foundation in algorithms and data.
                </p>
              </div>
            </div>
            <div className="timeline-item">
              <div className="timeline-content">
                <span className="date">Completed</span>
                <h3>Diploma in Information Technology</h3>
                <span className="institution">Gujarat Technological University</span>
                <p>Diploma in Information Technology with a strong foundation in programming, data structures,
                  and
                  software fundamentals.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="references" className="references-section section">
        <div className="references-noise"></div>
        <div className="references-container">
          <div className="references-header fade-in">
            <h2 className="section-title">References</h2>
            <p>Trusted by Visionary Teams</p>
            <p>
              Collaborating with industry leaders to deliver impactful, data-driven AI systems
              that redefine what's possible.
            </p>
          </div>
          <div className="references-track-wrapper fade-in">
            <div className="references-track">
              {/* Reference Cards */}
              <div className="reference-card">
                <div className="ref-card-top">
                  <svg className="ref-logo" viewBox="0 0 120 40" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="20" cy="20" r="12" fill="#111" />
                    <path d="M20 8L32 20L20 32" stroke="#fff" strokeWidth="2" />
                    <text x="42" y="26" fontFamily="var(--font-heading)" fontWeight="700" fontSize="18"
                      fill="#111" letterSpacing="-0.5">NTT DATA</text>
                  </svg>
                  <div className="ref-stars">
                    <i className="fas fa-star"></i><i className="fas fa-star"></i><i className="fas fa-star"></i><i
                      className="fas fa-star"></i><i className="fas fa-star"></i>
                  </div>
                </div>
                <p className="ref-text"><span>A motivated AI engineer skilled in building machine learning models
                  and production-ready backend systems.</span></p>
                <div className="ref-author">
                  <div className="ref-author-avatar">B</div>
                  <div className="ref-author-info">
                    <h4 className="ref-author-name">Brajesh Ashara</h4>
                    <p className="ref-author-role">Lead Consultant</p>
                  </div>
                </div>
              </div>
              {/* Card 2 */}
              <div className="reference-card">
                <div className="ref-card-top">
                  <svg className="ref-logo" viewBox="0 0 120 40" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect x="8" y="8" width="24" height="24" rx="6" fill="#111" />
                    <circle cx="20" cy="20" r="4" fill="#fff" />
                    <text x="42" y="26" fontFamily="var(--font-heading)" fontWeight="700" fontSize="18"
                      fill="#111" letterSpacing="-0.5">Reliance</text>
                  </svg>
                  <div className="ref-stars">
                    <i className="fas fa-star"></i><i className="fas fa-star"></i><i className="fas fa-star"></i><i
                      className="fas fa-star"></i><i className="fas fa-star"></i>
                  </div>
                </div>
                <p className="ref-text"><span>Consistently delivers high-quality work with strong expertise in data
                  insights and analytics dashboards.</span></p>
                <div className="ref-author">
                  <div className="ref-author-avatar">V</div>
                  <div className="ref-author-info">
                    <h4 className="ref-author-name">Vijay Dudhatra</h4>
                    <p className="ref-author-role">General Manager</p>
                  </div>
                </div>
              </div>
              {/* Card 3 */}
              <div className="reference-card">
                <div className="ref-card-top">
                  <svg className="ref-logo" viewBox="0 0 120 40" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M20 6L32 26H8L20 6Z" fill="#111" />
                    <text x="42" y="26" fontFamily="var(--font-heading)" fontWeight="700" fontSize="18"
                      fill="#111" letterSpacing="-0.5">AplombSoft</text>
                  </svg>
                  <div className="ref-stars">
                    <i className="fas fa-star"></i><i className="fas fa-star"></i><i className="fas fa-star"></i><i
                      className="fas fa-star"></i><i className="fas fa-star"></i>
                  </div>
                </div>
                <p className="ref-text"><span>Brings strong technical knowledge and delivers reliable,
                  well-structured solutions.</span></p>
                <div className="ref-author">
                  <div className="ref-author-avatar">M</div>
                  <div className="ref-author-info">
                    <h4 className="ref-author-name">Mihir Nirmal</h4>
                    <p className="ref-author-role">Lead Engineer</p>
                  </div>
                </div>
              </div>
              {/* Card 4 */}
              <div className="reference-card">
                <div className="ref-card-top">
                  <svg className="ref-logo" viewBox="0 0 120 40" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M20 6L32 26H8L20 6Z" fill="#111" />
                    <text x="42" y="26" fontFamily="var(--font-heading)" fontWeight="700" fontSize="18"
                      fill="#111" letterSpacing="-0.5">Exillar</text>
                  </svg>
                  <div className="ref-stars">
                    <i className="fas fa-star"></i><i className="fas fa-star"></i><i className="fas fa-star"></i><i
                      className="fas fa-star"></i><i className="fas fa-star"></i>
                  </div>
                </div>
                <p className="ref-text"><span>An exceptional engineer with deep expertise in AI architectures and scalable data pipelines.</span></p>
                <div className="ref-author">
                  <div className="ref-author-avatar">H</div>
                  <div className="ref-author-info">
                    <h4 className="ref-author-name">Harsh Mer</h4>
                    <p className="ref-author-role">Lead AI Engineer</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <FAQSection />

      <section id="home-contact" className="section">
        <div className="container">
          <div className="contact-wrapper">
            <h2 className="section-title">Contact Me</h2>
            <p className="contact-subtitle">Ready to collaborate or discuss AI solutions?</p>
            <Link to="/contact" className="btn-primary">
              Get in Touch
            </Link>
          </div>
        </div>
      </section>
    </>
  );
};

export default Home;
