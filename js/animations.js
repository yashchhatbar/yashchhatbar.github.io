(() => {
  const prefersReducedMotion = () =>
    window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const GREETINGS = [
    "नमस्ते",
    "こんにちは",
    "BONJOUR",
    "HOLA",
    "CIAO",
    "नमस्कार",
    "ನಮಸ್ಕಾರ",
    "GUTEN TAG",
    "HELLO",
    "શુભ દિવસ",
  ];

  // Full-screen intro overlay
  const initIntroOverlay = () => {
    const overlay = document.getElementById("intro-overlay");
    const wordEl = document.getElementById("intro-word");
    if (!overlay || !wordEl) return;

    if (prefersReducedMotion()) {
      overlay.remove();
      return;
    }

    document.body.style.overflow = "hidden";

    let index = 0;
    const wordDuration = 200;
    const fadeDuration = 200;

    const showNextWord = () => {
      if (index >= GREETINGS.length) {
        overlay.style.transition = "opacity 0.6s ease";
        overlay.style.opacity = "0";
        setTimeout(() => {
          overlay.remove();
          document.body.style.overflow = "";
        }, 600);
        return;
      }

      wordEl.textContent = GREETINGS[index];
      wordEl.style.opacity = "1";

      setTimeout(() => {
        wordEl.style.opacity = "0";
        index += 1;
        setTimeout(showNextWord, fadeDuration);
      }, wordDuration);
    };

    setTimeout(showNextWord, 100);
  };

  const initPreloader = () => {
    const preloader = document.getElementById('preloader');
    const wordEl = document.getElementById('preloader-word');

    // If preloader elements don't exist on page, exit early
    if (!preloader || !wordEl) return;

    // Prevent scrolling while preloader
    document.body.style.overflow = 'hidden';
    window.scrollTo(0, 0);

    // Greetings array (international)
    const greetings = [
      "नमस्ते",
      "こんにちは",
      "BONJOUR",
      "HOLA",
      "CIAO",
      "नमस्कार",
      "ನಮಸ್ಕಾರ",
      "GUTEN TAG",
      "HELLO",
      "શુભ દિવસ",
      ];

    const TIMING_IN = 250;  // How fast word appears
    const TIMING_OUT = 200; // How fast word disappears
    let currentIndex = 0;

    const animateGreeting = () => {
        if (currentIndex >= greetings.length) {
            // All greetings shown, fade out preloader
            endPreloader();
            return;
        }

        // Set text
        wordEl.textContent = greetings[currentIndex];

        // Remove exit class, add active to trigger CSS transition
        wordEl.classList.remove('exit');
        wordEl.classList.add('active');

        // Wait then exit
        setTimeout(() => {
            wordEl.classList.remove('active');
            wordEl.classList.add('exit');

            currentIndex++;

            // Wait for exit transition, then show next
            setTimeout(animateGreeting, TIMING_OUT);
        }, TIMING_IN);
    };

    const endPreloader = () => {
        // Hide preloader overlay smoothly
        preloader.classList.add('preloader-hidden');

        // Allow scrolling again
        document.body.style.overflow = '';

        // Remove from DOM eventually for clean tree
        setTimeout(() => {
            preloader.remove();
        }, 800);
    };

    // Start sequence after short initial delay
    setTimeout(animateGreeting, 100);
};

// --- Execute on DOM Ready ---
document.addEventListener('DOMContentLoaded', () => {
    initPreloader();
    if (typeof initScrollAnimations === "function") {
      initScrollAnimations();
    }
});

  // Scroll reveal animations
  const initScrollReveal = () => {
    if (prefersReducedMotion()) return;

    const observerOptions = { threshold: 0.1, rootMargin: "0px 0px -50px 0px" };
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("visible");
        observer.unobserve(entry.target);
      });
    }, observerOptions);

    // Staggered grids/timelines
    const staggerContainers = document.querySelectorAll(".skills-grid, .projects-grid, .timeline");
    staggerContainers.forEach((container) => {
      Array.from(container.children).forEach((child, index) => {
        const delay = ((index % 5) + 1) * 100; // 100..500
        child.classList.add(`delay-${delay}`);
        child.classList.add("hidden");
        observer.observe(child);
      });
    });

    const animatedElements = document.querySelectorAll(
      ".section-title, .about-text, .hero-content > *, .contact-wrapper > *, .references-header, .references-track-wrapper"
    );
    animatedElements.forEach((el) => {
      el.classList.add("hidden");
      observer.observe(el);
    });

    document.querySelectorAll(".section-title").forEach((title) => {
      title.classList.remove("hidden");
      title.classList.add("hidden-left");
      observer.observe(title);
    });
  };

  // Footer greeting scramble (index)
  const initFooterGreeting = () => {
    const el = document.getElementById("layerIn");
    if (!el) return;
    if (prefersReducedMotion()) return;

    const POOL = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    const rnd = () => POOL[Math.floor(Math.random() * POOL.length)];

    let current = 0;
    let busy = false;

    const scrambleReveal = (target, ms = 500) =>
      new Promise((resolve) => {
        const steps = 22;
        const stepMs = ms / steps;
        let step = 0;

        const id = setInterval(() => {
          step += 1;
          const locked = Math.round((step / steps) * target.length);
          let out = "";
          for (let i = 0; i < target.length; i += 1) {
            out += i < locked ? target[i] : rnd();
          }
          el.textContent = out;
          if (step >= steps) {
            clearInterval(id);
            el.textContent = target;
            resolve();
          }
        }, stepMs);
      });

    const nextWord = async () => {
      if (busy) return;
      busy = true;

      const next = (current + 1) % GREETINGS.length;
      const nextText = GREETINGS[next];

      await scrambleReveal(nextText, 300);
      current = next;
      busy = false;
    };

    setInterval(nextWord, 1200);
  };

  // Remove duplicated reference cards from HTML by cloning at runtime (seamless marquee)
  const initReferenceTrackClones = () => {
    const track = document.querySelector(".references-track");
    if (!track) return;
    if (track.dataset.cloned === "true") return;

    const cards = Array.from(track.children);
    if (!cards.length) return;

    cards.forEach((card) => track.appendChild(card.cloneNode(true)));
    track.dataset.cloned = "true";
  };

  const onReady = (fn) => {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", fn, { once: true });
      return;
    }
    fn();
  };

  onReady(() => {
    initIntroOverlay();
    initScrollReveal();
    initFooterGreeting();
    initReferenceTrackClones();
  });
})();

