import { Link } from "react-router-dom"

function ProjectCard({ project }) {

    return (

        <Link
            to={`/projects/${project.slug}`}
            className="project-card fade-in"
        >

            <div className="project-image-wrapper">

                <img
                    src={project.image}
                    alt={project.title}
                />

            </div>

            <div className="project-footer">

                <h3>{project.title}</h3>

                <span className="project-arrow">
                    →
                </span>

            </div>

        </Link>

    )

}

export default ProjectCard