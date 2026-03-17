import { useState } from "react";

const faqs = [
    {
        q: "What kind of problems do you enjoy solving?",
        a: "I enjoy solving problems that involve data-driven decision making, intelligent automation, and scalable backend systems. My focus is on building solutions where artificial intelligence and software engineering can simplify complex processes and create measurable impact."
    },
    {
        q: "How do you approach building AI-powered applications?",
        a: "My process usually starts with understanding the problem and the available data, followed by experimenting with different machine learning models and evaluating their performance. Once the best approach is identified, I integrate the model into a production-ready backend system using APIs so it can be used in real applications."
    },
    {
        q: "What makes your development approach different?",
        a: "I focus on practical and scalable solutions rather than just prototypes. My goal is to build systems that are technically sound, maintainable, efficient, and useful in real-world scenarios."
    },
    {
        q: "What are you currently learning or exploring?",
        a: "I continuously explore advanced machine learning techniques, data engineering workflows, and scalable backend architectures."
    },
    {
        q: "How can companies or collaborators work with you?",
        a: "I am open to internships, collaborations, and project-based work where I can contribute to building AI solutions, data-driven applications, or backend systems."
    }
];

export default function FAQSection() {
    const [openIndex, setOpenIndex] = useState(0);

    const toggle = (index) => {
        setOpenIndex(openIndex === index ? null : index);
    };
    return (
        <section id="faq" className="container-fluid faq_section section">
            <div className="container">
                <div className="faq_section_text">
                    <h2 className="section-title">Frequently asked questions</h2>

                    <div className="accordion-faq_section">

                        {faqs.map((faq, index) => (
                            <div
                                key={index}
                                className={`accordion-box ${openIndex === index ? "open" : ""}`}
                            >

                                <div
                                    className="accordion-heading"
                                    onClick={() => toggle(index)}
                                >
                                    {faq.q}
                                </div>

                                {openIndex === index && (
                                    <div className="accordion_body">
                                        {faq.a}
                                    </div>
                                )}

                            </div>
                        ))}

                    </div>
                </div>
            </div>
        </section>
    );
}