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
  update();
};

const initResumeDownload = () => {
  document.addEventListener('click', (e) => {
    const link = e.target.closest('a[download][href$=".pdf"]');
    if (!link) return;

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
      from_name: form.querySelector("#from_name")?.value || "",
      from_email: form.querySelector("#from_email")?.value || "",
      message: form.querySelector("#message")?.value || "",
      origin: window.location.href,
    };

    const showSuccess = () => {
      form.style.display = "none";
      successMsg.style.display = "block";
      successMsg.scrollIntoView({ behavior: "smooth", block: "center" });
    };

    try {
      if (!window.emailjs || typeof window.emailjs.send !== "function") {
        throw new Error("EmailJS not loaded");
      }

      await window.emailjs.send(SERVICE_ID, TEMPLATE_ID_OWNER, templateParams);

      if (TEMPLATE_ID_AUTO_REPLY) {
        window.emailjs.send(SERVICE_ID, TEMPLATE_ID_AUTO_REPLY, templateParams)
          .catch(err => { });
      }

      showSuccess();
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

export const initApp = () => {
  initLucideIcons();
  initNavbarScrolledState();
  initResumeDownload();
  initEmailForm();
};
