let cursorAnimationFrame;

export const initCursor = () => {
  const cursorDot = document.querySelector(".cursor-dot");
  const cursorOutline = document.querySelector(".cursor-outline");
  if (!cursorDot || !cursorOutline) return;

  if (!window.matchMedia("(pointer: fine)").matches) return;

  // Cleanup previous animation if it's running
  if (cursorAnimationFrame) cancelAnimationFrame(cursorAnimationFrame);

  let mouseX = 0;
  let mouseY = 0;
  let outlineX = 0;
  let outlineY = 0;
  let isFirstMove = true;

  const onMouseMove = (e) => {
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
  };

  document.addEventListener("mousemove", onMouseMove, { passive: true });

  const animate = () => {
    const speed = 0.15;
    outlineX += (mouseX - outlineX) * speed;
    outlineY += (mouseY - outlineY) * speed;
    cursorOutline.style.left = `${outlineX}px`;
    cursorOutline.style.top = `${outlineY}px`;
    cursorAnimationFrame = requestAnimationFrame(animate);
  };
  animate();

  const isHoverTarget = (target) =>
    !!target.closest("a, button, .project-card, input, textarea, .logo, .theme-toggle");

  const onMouseOver = (e) => {
    if (isHoverTarget(e.target)) document.body.classList.add("cursor-hover");
  };

  const onMouseOut = (e) => {
    if (isHoverTarget(e.target)) document.body.classList.remove("cursor-hover");
  };

  const onMouseDown = () => document.body.classList.add("cursor-active");
  const onMouseUp = () => document.body.classList.remove("cursor-active");

  document.addEventListener("mouseover", onMouseOver, { passive: true });
  document.addEventListener("mouseout", onMouseOut, { passive: true });
  document.addEventListener("mousedown", onMouseDown);
  document.addEventListener("mouseup", onMouseUp);

  // Return cleanup function if needed
  return () => {
    document.removeEventListener("mousemove", onMouseMove);
    document.removeEventListener("mouseover", onMouseOver);
    document.removeEventListener("mouseout", onMouseOut);
    document.removeEventListener("mousedown", onMouseDown);
    document.removeEventListener("mouseup", onMouseUp);
    if (cursorAnimationFrame) cancelAnimationFrame(cursorAnimationFrame);
  };
};

