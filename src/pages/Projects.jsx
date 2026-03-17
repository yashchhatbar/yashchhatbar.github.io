import React, { useEffect } from 'react';
import "../css/project.css"

import { projects } from "../data/projectsData"
import ProjectCard from "../components/ProjectCard"

const Projects = () => {

  useEffect(() => {
    if (window.lucide) {
      window.lucide.createIcons();
    }
  }, []);

  return (
    <>
      <header className="projects-header">
        <h1>
          All <span className="gradient-text">Projects</span>
        </h1>

        <p>
          A comprehensive archive of models, analytics dashboards,
          and engineered AI systems.
        </p>
      </header>

      <section id="all-projects" className="section">

        <div className="container">

          <h2 className="section-title">All Projects</h2>

          <div className="projects-grid projects-grid--grid">

            {projects.map((project) => (

              <ProjectCard
                key={project.slug}
                project={project}
              />

            ))}

          </div>

        </div>

      </section>
    </>
  );
};

export default Projects;