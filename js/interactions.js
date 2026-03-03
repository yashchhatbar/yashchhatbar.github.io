(() => {
  const onReady = (fn) => {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", fn, { once: true });
      return;
    }
    fn();
  };

  const initCursor = () => {
    const cursorDot = document.querySelector(".cursor-dot");
    const cursorOutline = document.querySelector(".cursor-outline");
    if (!cursorDot || !cursorOutline) return;

    if (!window.matchMedia("(pointer: fine)").matches) return;

    let mouseX = 0;
    let mouseY = 0;
    let outlineX = 0;
    let outlineY = 0;
    let isFirstMove = true;

    document.addEventListener(
      "mousemove",
      (e) => {
        mouseX = e.clientX;
        mouseY = e.clientY;

        cursorDot.style.left = `${mouseX}px`;
        cursorDot.style.top = `${mouseY}px`;

        if (isFirstMove) {
          outlineX = mouseX;
          outlineY = mouseY;
          isFirstMove = false;
          cursorDot.style.opacity = "1";
          cursorOutline.style.opacity = "1";
        }
      },
      { passive: true }
    );

    const animate = () => {
      const speed = 0.15;
      outlineX += (mouseX - outlineX) * speed;
      outlineY += (mouseY - outlineY) * speed;
      cursorOutline.style.left = `${outlineX}px`;
      cursorOutline.style.top = `${outlineY}px`;
      requestAnimationFrame(animate);
    };
    animate();

    const isHoverTarget = (target) =>
      !!target.closest("a, button, .project-card, input, textarea, .logo, .theme-toggle");

    document.addEventListener(
      "mouseover",
      (e) => {
        if (isHoverTarget(e.target)) document.body.classList.add("cursor-hover");
      },
      { passive: true }
    );

    document.addEventListener(
      "mouseout",
      (e) => {
        if (isHoverTarget(e.target)) document.body.classList.remove("cursor-hover");
      },
      { passive: true }
    );

    document.addEventListener("mousedown", () => document.body.classList.add("cursor-active"));
    document.addEventListener("mouseup", () => document.body.classList.remove("cursor-active"));
  };

  const initMobileMenu = () => {
    const hamburger = document.querySelector(".hamburger");
    const navLinksContainer = document.querySelector(".nav-links");
    const icon = hamburger ? hamburger.querySelector("i") : null;
    if (!hamburger || !navLinksContainer) return;

    const setExpanded = (expanded) => {
      hamburger.setAttribute("aria-expanded", expanded ? "true" : "false");
    };

    const open = () => {
      navLinksContainer.classList.add("active");
      document.body.classList.add("menu-open");
      setExpanded(true);
      if (icon) {
        icon.classList.add("fa-times");
        icon.classList.remove("fa-bars");
      }
    };

    const close = () => {
      navLinksContainer.classList.remove("active");
      document.body.classList.remove("menu-open");
      setExpanded(false);
      if (icon) {
        icon.classList.remove("fa-times");
        icon.classList.add("fa-bars");
      }
    };

    hamburger.addEventListener("click", () => {
      if (navLinksContainer.classList.contains("active")) close();
      else open();
    });

    navLinksContainer.addEventListener("click", (e) => {
      const link = e.target.closest("a");
      if (!link) return;
      if (window.innerWidth <= 1024) close();
    });

    window.addEventListener("resize", () => {
      if (window.innerWidth > 1024) close();
    });
  };

  onReady(() => {
    initCursor();
    initMobileMenu();
  });
})();

