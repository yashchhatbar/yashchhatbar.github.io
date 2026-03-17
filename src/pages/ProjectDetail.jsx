import { useParams, Link } from "react-router-dom";
import { projects } from "../data/projectsData";
import "../css/projectDetail.css";

const ProjectDetail = () => {

    const { slug } = useParams();

    const project = projects.find(p => p.slug === slug);

    if (!project) {
        return <div style={{ padding: "120px" }}>Project not found</div>;
    }

    return (

        <div className="project-detail-page">

            {/* HERO IMAGE FULL SCREEN */}

            <section className="full-bleed hero-section">

                <img
                    src={project.hero}
                    alt={project.title}
                    className="hero-image"
                />

            </section>


            {/* CONTENT */}

            <div className="project-container">

                <Link to="/projects" className="back-btn">
                    <i data-lucide="arrow-left" width="18" height="18"></i>
                </Link>
                <span className="project-year">
                    {project.year}
                </span>

                <h1 className="project-title">
                    {project.title}
                </h1>

                <div className="tech-stack">
                    {project.tech.map((tech, i) => (
                        <span key={i} className="tech-pill">{tech}</span>
                    ))}
                </div>

                <div className="case-section">

                    <h2>Problem</h2>
                    <div className="case-text">
                        {project.problem.trim().split("\n\n").map((paragraph, index) => (
                            <p key={index}>{paragraph.trim()}</p>
                        ))}
                    </div>

                    <h2>Solution</h2>
                    <div className="case-text">
                        {project.solution.trim().split("\n\n").map((paragraph, index) => (
                            <p key={index}>{paragraph.trim()}</p>
                        ))}
                    </div>

                    <h2>Impact</h2>
                    <div className="case-text">
                        {project.impact.trim().split("\n\n").map((paragraph, index) => (
                            <p key={index}>{paragraph.trim()}</p>
                        ))}
                    </div>

                </div>

                <a
                    href={project.github}
                    target="_blank"
                    className="github-btn"
                >
                    View on GitHub
                </a>

            </div>


            {/* FULL WIDTH GALLERY */}

            {project.gallery.map((img, i) => (

                <section key={i} className="full-bleed gallery-section">

                    <img
                        src={img}
                        alt="project"
                        className="gallery-image"
                    />

                </section>

            ))}

        </div>

    );

};

export default ProjectDetail;