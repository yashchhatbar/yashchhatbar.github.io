(() => {
  const onReady = (fn) => {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", fn, { once: true });
      return;
    }
    fn();
  };

  const isIndexPage = () => {
    const page = (location.pathname.split("/").pop() || "index.html").toLowerCase();
    return page === "index.html";
  };

  const initLucideIcons = () => {
    if (window.lucide && typeof window.lucide.createIcons === "function") {
      window.lucide.createIcons();
    }
  };

  const initNavbarScrolledState = () => {
    const navbar = document.querySelector(".navbar");
    if (!navbar) return;

    const update = () => {
      if (window.scrollY > 50) navbar.classList.add("scrolled");
      else navbar.classList.remove("scrolled");
    };

    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("load", update, { once: true });
    update();
  };

  const initActiveNav = () => {
    if (!isIndexPage()) return;

    const sections = document.querySelectorAll("section, header.hero");
    const navItems = document.querySelectorAll(".nav-links a");
    if (!sections.length || !navItems.length) return;

    const updateActiveNav = () => {
      let currentSectionId = "";
      const viewLine = window.innerHeight * 0.25;

      sections.forEach((section) => {
        const rect = section.getBoundingClientRect();
        if (rect.top <= viewLine && rect.bottom >= viewLine) {
          currentSectionId = section.getAttribute("id") || "";
          if (!currentSectionId && section.classList.contains("hero")) currentSectionId = "hero";
        }
      });

      if (window.scrollY < 100) currentSectionId = "hero";

      if (window.innerHeight + window.scrollY >= document.body.offsetHeight - 50) {
        const last = sections[sections.length - 1];
        if (last) currentSectionId = last.getAttribute("id") || currentSectionId;
      }

      navItems.forEach((item) => {
        item.classList.remove("active");
        const href = item.getAttribute("href") || "";
        if (!currentSectionId) return;
        if (href === `#${currentSectionId}` || href.endsWith(`#${currentSectionId}`)) {
          item.classList.add("active");
        }
      });
    };

    window.addEventListener("scroll", updateActiveNav, { passive: true });
    window.addEventListener("load", updateActiveNav, { once: true });
    updateActiveNav();
  };

  const initAccordion = () => {
    const headings = document.querySelectorAll(".accordion-heading");
    if (!headings.length) return;

    headings.forEach((heading) => {
      heading.addEventListener("click", () => {
        const currentBox = heading.closest(".accordion-box");
        if (!currentBox) return;
        const isOpen = currentBox.classList.contains("open");

        document.querySelectorAll(".accordion-box.open").forEach((box) => box.classList.remove("open"));
        if (!isOpen) currentBox.classList.add("open");
      });
    });
  };

  const initResumeDownload = () => {
    const downloadLinks = document.querySelectorAll('a[download][href$=".pdf"]');
    if (!downloadLinks.length) return;

    downloadLinks.forEach((link) => {
      link.addEventListener("click", (e) => {
        e.preventDefault();

        const url = link.getAttribute("href");
        if (!url) return;
        const filename = link.getAttribute("download") || "Yash_Chhatbar_Resume.pdf";

        fetch(url)
          .then((response) => response.blob())
          .then((blob) => {
            const blobUrl = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.style.display = "none";
            a.href = blobUrl;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(blobUrl);
            document.body.removeChild(a);
          })
          .catch((error) => {
            console.error("Error downloading the resume:", error);
            window.open(url, "_blank", "noopener,noreferrer");
          });
      });
    });
  };

  const initEmailForm = () => {
    const form = document.getElementById("contactFormMain");
    if (!form) return;

    const PUBLIC_KEY = "AVb4JrM16z6xk21t4";
    const SERVICE_ID = "service_0bsyuc4";
    const TEMPLATE_ID_OWNER = "template_wgy0ifq";
    const TEMPLATE_ID_AUTO_REPLY = "template_62qr6mp";

    const btn = document.getElementById("submitBtn");
    const successMsg = document.getElementById("successMessage");
    const errorMsg = document.getElementById("errorMessage");
    const originField = document.getElementById("origin_field");

    if (window.emailjs && typeof window.emailjs.init === "function") {
      window.emailjs.init(PUBLIC_KEY);
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!btn || !successMsg || !errorMsg) return;

      if (originField) originField.value = window.location.href;

      const originalBtnText = btn.innerText;
      btn.innerText = "Sending...";
      btn.disabled = true;
      errorMsg.style.display = "none";

      const templateParams = {
        from_name: /** @type {HTMLInputElement|null} */ (document.getElementById("from_name"))?.value || "",
        from_email: /** @type {HTMLInputElement|null} */ (document.getElementById("from_email"))?.value || "",
        message: /** @type {HTMLTextAreaElement|null} */ (document.getElementById("message"))?.value || "",
        origin: window.location.href,
      };

      const showSuccess = () => {
        form.style.display = "none";
        successMsg.style.display = "block";
        successMsg.scrollIntoView({ behavior: "smooth", block: "center" });
      };

      try {
        // Keep current UX: show success quickly, send in background.
        setTimeout(showSuccess, 300);

        if (!window.emailjs || typeof window.emailjs.send !== "function") {
          throw new Error("EmailJS not loaded");
        }

        window.emailjs
          .send(SERVICE_ID, TEMPLATE_ID_OWNER, templateParams)
          .then(() => console.log("Owner email sent"))
          .catch((err) => console.error("Owner email failed:", err));

        if (TEMPLATE_ID_AUTO_REPLY) {
          window.emailjs
            .send(SERVICE_ID, TEMPLATE_ID_AUTO_REPLY, templateParams)
            .then(() => console.log("Auto reply sent"))
            .catch((err) => console.error("Auto reply failed:", err));
        }

        form.reset();
      } catch (error) {
        console.error("Unexpected Error:", error);
        errorMsg.innerText = "Failed to send message. Please try again later.";
        errorMsg.style.display = "block";
        btn.innerText = originalBtnText;
        btn.disabled = false;
      }
    });
  };

  onReady(() => {
    initLucideIcons();
    initNavbarScrolledState();
    initActiveNav();
    initAccordion();
    initResumeDownload();
    initEmailForm();
  });


})();
