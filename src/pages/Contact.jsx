import React, { useEffect } from 'react';

const Contact = () => {
  useEffect(() => {
    // Re-initialize Lucide icons
    if (window.lucide) {
      window.lucide.createIcons();
    }

    // Initialize EmailJS if needed (though it's in main.js)
    // Dispatch event for external scripts
    window.dispatchEvent(new Event('DOMContentLoaded'));
  }, []);

  return (
    <>
      <div className="navbar-spacer" aria-hidden="true"></div>

      <section className="section">
        <div className="container">
          <div className="contact-page-container">
            {/* Left Side: Contact Info */}
            <div className="contact-info-panel">
              <h1 className="section-title">Get in Touch</h1>
              <p className="contact-lead">
                I am currently open to new opportunities in AI & Machine Learning. Whether you have a question,
                a project idea, or just want to say hi, I’ll appreciate the connection.
              </p>

              <div className="info-item">
                <span className="info-label">Email</span>
                <div className="info-value">
                  <a href="mailto:yashchhatbar11@gmail.com">yashchhatbar11@gmail.com</a>
                </div>
              </div>

              <div className="info-item">
                <span className="info-label">Social Highlights</span>
                <span className="info-value">
                  <a href="https://github.com/yashchhatbar" target="_blank" rel="noopener noreferrer"
                    style={{ marginRight: '15px' }}><i className="fab fa-github"></i> GitHub</a>
                  <a href="https://www.linkedin.com/in/yashchhatbar" target="_blank"
                    rel="noopener noreferrer"><i className="fab fa-linkedin"></i> LinkedIn</a>
                </span>
              </div>

              <div className="info-item">
                <span className="info-label">Location</span>
                <span className="info-value"><a href="https://share.google/forIQR9ApT5REuLOs" target="_blank"
                    rel="noopener noreferrer"><i class="fa fa-map-marker" aria-hidden="true"></i>Ahmedabad, Gujarat, India</a></span>
              </div>
            </div>

            {/* Right Side: Contact Form */}
            <div className="contact-form-panel">
              <form id="contactFormMain">
                <input type="hidden" name="to_name" value="Yash Chhatbar" />
                <input type="hidden" name="origin" id="origin_field" />

                <label className="form-group">
                  <span className="form-label">Full Name</span>
                  <input type="text" id="from_name" name="from_name" className="form-input"
                    placeholder="Elon Musk" required autoComplete="name" />
                </label>

                <label className="form-group">
                  <span className="form-label">Email Address</span>
                  <input type="email" id="from_email" name="from_email" className="form-input"
                    placeholder="elon@tesla.com" required autoComplete="email" />
                </label>

                <label className="form-group">
                  <span className="form-label">Message</span>
                  <textarea id="message" name="message" className="form-textarea"
                    placeholder="I'd love to chat about a potential AI integration..." required></textarea>
                </label>

                <p className="form-error" id="errorMessage" aria-live="polite">We encountered an error sending
                  your message. Please email me directly.</p>

                <button type="submit" className="btn-primary btn-block" id="submitBtn">
                  Send Message <i className="fas fa-paper-plane"></i>
                </button>
              </form>

              <div className="form-success" id="successMessage" style={{ display: 'none' }} aria-live="polite">
                <i className="fas fa-check-circle form-success-icon"></i>
                <h3>Message Sent!</h3>
                <p>Thank you for reaching out. I'll get back to you within 24 hours.</p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </>
  );
};

export default Contact;
