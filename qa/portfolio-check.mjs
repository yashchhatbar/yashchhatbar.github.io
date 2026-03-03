import { chromium } from "playwright";

const BASE_URL = "http://localhost:8000";
const VIEWPORTS = [
  { width: 1920, height: 900 },
  { width: 1440, height: 900 },
  { width: 768, height: 900 },
  { width: 480, height: 900 },
];

const PAGES = [
  { name: "index.html", path: "/index.html" },
  { name: "projects.html", path: "/projects.html" },
  { name: "contact.html", path: "/contact.html" },
];

function uniq(arr) {
  return Array.from(new Set(arr));
}

function truncate(s, n = 160) {
  if (!s) return s;
  return s.length > n ? `${s.slice(0, n)}…` : s;
}

async function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function attachCollectors(page) {
  /** @type {{type:string, text:string, location?:string}[]} */
  const consoleErrors = [];
  /** @type {string[]} */
  const pageErrors = [];
  /** @type {{url:string, method:string, failure?:string, resourceType?:string}[]} */
  const requestFailures = [];

  page.on("console", async (msg) => {
    if (msg.type() !== "error") return;
    const loc = msg.location?.();
    const location = loc && loc.url ? `${loc.url}:${loc.lineNumber || 0}:${loc.columnNumber || 0}` : undefined;
    consoleErrors.push({ type: msg.type(), text: msg.text(), location });
  });
  page.on("pageerror", (err) => {
    pageErrors.push(String(err && err.stack ? err.stack : err));
  });
  page.on("requestfailed", (req) => {
    requestFailures.push({
      url: req.url(),
      method: req.method(),
      failure: req.failure()?.errorText,
      resourceType: req.resourceType(),
    });
  });

  return { consoleErrors, pageErrors, requestFailures };
}

async function addEmailJsStubRouting(context) {
  // Ensure clicking "Send Message" does not send real emails and does not log console errors.
  const stub = String.raw`(() => {
    const api = {
      init: () => {},
      send: () => Promise.resolve({ status: 200, text: "OK (stubbed)" })
    };
    Object.defineProperty(window, "emailjs", { value: api, writable: false, configurable: true });
  })();`;

  await context.route("**/@emailjs/browser@3/dist/email.min.js", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/javascript; charset=utf-8",
      body: stub,
    });
  });
}

async function checkHorizontalOverflow(page) {
  return await page.evaluate(() => {
    const w = window.innerWidth;
    const sw = Math.max(document.documentElement.scrollWidth, document.body ? document.body.scrollWidth : 0);
    const hasOverflow = sw > w + 1;
    if (!hasOverflow) return { hasOverflow: false, scrollWidth: sw, innerWidth: w, offenders: [] };

    const offenders = [];
    const els = Array.from(document.querySelectorAll("*"));
    for (const el of els) {
      const rect = el.getBoundingClientRect();
      const leftOverflow = Math.max(0, 0 - rect.left);
      const rightOverflow = Math.max(0, rect.right - w);
      const amt = Math.max(leftOverflow, rightOverflow);
      if (amt <= 1) continue;

      const tag = el.tagName.toLowerCase();
      const id = el.id ? `#${el.id}` : "";
      const cls = el.classList && el.classList.length ? `.${Array.from(el.classList).slice(0, 3).join(".")}` : "";
      const sel = `${tag}${id}${cls}`;

      offenders.push({ selector: sel, overflowPx: Math.round(amt) });
      if (offenders.length >= 8) break;
    }

    offenders.sort((a, b) => b.overflowPx - a.overflowPx);
    return { hasOverflow: true, scrollWidth: sw, innerWidth: w, offenders };
  });
}

async function checkInPageAnchors(page) {
  return await page.evaluate(() => {
    const links = Array.from(document.querySelectorAll("a[href]"))
      .map((a) => a.getAttribute("href") || "")
      .filter((h) => h.startsWith("#"));

    const broken = [];
    const placeholders = [];
    for (const href of links) {
      if (href === "#" || href === "#0") {
        placeholders.push(href);
        continue;
      }
      const id = href.slice(1);
      const target = document.getElementById(id) || document.querySelector(`[name="${CSS.escape(id)}"]`);
      if (!target) broken.push(href);
    }
    return { broken: Array.from(new Set(broken)), placeholders: Array.from(new Set(placeholders)) };
  });
}

async function checkLocalLinks(page) {
  const hrefs = await page.evaluate(() => {
    const out = [];
    for (const a of Array.from(document.querySelectorAll("a[href]"))) {
      const href = (a.getAttribute("href") || "").trim();
      if (!href) continue;
      if (href.startsWith("#")) continue;
      if (href.startsWith("mailto:") || href.startsWith("tel:")) continue;
      if (href.startsWith("javascript:")) continue;
      if (href.startsWith("http://") || href.startsWith("https://")) continue;
      out.push(href);
    }
    return Array.from(new Set(out));
  });

  const local = [];
  const placeholders = [];
  for (const href of hrefs) {
    if (href === "#") placeholders.push(href);
    else local.push(href);
  }

  const results = [];
  for (const href of local.slice(0, 30)) {
    const url = href.startsWith("/") ? `${BASE_URL}${href}` : `${BASE_URL}/${href}`;
    try {
      const resp = await page.request.get(url, { timeout: 8000 });
      results.push({ href, url, status: resp.status() });
    } catch (e) {
      results.push({ href, url, status: 0, error: String(e) });
    }
  }

  return { checked: results, placeholderHrefs: uniq(placeholders) };
}

async function checkHamburger(page, viewportWidth) {
  if (viewportWidth > 1024) return { checked: false };

  const hamburger = page.locator(".hamburger");
  const navLinks = page.locator(".nav-links");
  const exists = (await hamburger.count()) > 0 && (await navLinks.count()) > 0;
  if (!exists) return { checked: true, ok: false, issue: "Missing .hamburger or .nav-links" };

  const box = await hamburger.boundingBox();
  const visible = box !== null;
  if (!visible) return { checked: true, ok: false, issue: "Hamburger not visible at this width" };

  const ariaBefore = await hamburger.getAttribute("aria-expanded");
  await hamburger.click();
  await page.waitForTimeout(200);
  const navActive = await navLinks.evaluate((el) => el.classList.contains("active"));
  const bodyMenuOpen = await page.evaluate(() => document.body.classList.contains("menu-open"));
  const ariaAfter = await hamburger.getAttribute("aria-expanded");

  await hamburger.click();
  await page.waitForTimeout(150);
  const navClosed = !(await navLinks.evaluate((el) => el.classList.contains("active")));

  const ok = navActive && bodyMenuOpen && ariaAfter === "true" && navClosed;
  return {
    checked: true,
    ok,
    details: { ariaBefore, ariaAfter, navActive, bodyMenuOpen, navClosed },
    issue: ok ? undefined : "Hamburger toggle did not correctly open/close nav",
  };
}

async function checkIndexBehaviors(page, viewportWidth) {
  const issues = [];

  // Intro overlay should disappear.
  const overlayInitially = (await page.locator("#intro-overlay").count()) > 0;
  if (overlayInitially) {
    try {
      await page.waitForSelector("#intro-overlay", { state: "detached", timeout: 9000 });
    } catch {
      issues.push("Intro overlay (#intro-overlay) did not disappear within 9s");
    }
    const overflow = await page.evaluate(() => getComputedStyle(document.body).overflow);
    if (overflow === "hidden") issues.push("Body overflow remained hidden after intro overlay");
  }

  // References marquee animates + clones created.
  const track = page.locator(".references-track");
  if ((await track.count()) === 0) {
    issues.push("Missing .references-track");
  } else {
    const prefersReduced = await page.evaluate(() => window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches);
    const childCount = await track.evaluate((el) => el.children.length);
    if (childCount < 8) issues.push(`References track did not clone cards (children=${childCount}, expected >= 8)`);

    if (!prefersReduced) {
      const a0 = await track.evaluate((el) => getComputedStyle(el).animationName);
      const t0 = await track.evaluate((el) => getComputedStyle(el).transform);
      await page.waitForTimeout(900);
      const t1 = await track.evaluate((el) => getComputedStyle(el).transform);
      if (!a0 || a0 === "none") issues.push("References track has no CSS animation (animation-name: none)");
      else if (t0 === t1) issues.push("References track transform did not change (marquee may not be animating)");
    }
  }

  // FAQ accordion toggles.
  const headings = page.locator(".accordion-heading");
  if ((await headings.count()) >= 2) {
    await headings.nth(1).click();
    await page.waitForTimeout(150);
    const secondOpen = await headings
      .nth(1)
      .locator("..")
      .evaluate((el) => el.closest(".accordion-box")?.classList.contains("open"));
    if (!secondOpen) issues.push("FAQ accordion did not open on click (2nd item)");
  } else {
    issues.push("FAQ accordion headings not found");
  }

  // Footer greeting updates (index only: #layerIn).
  const layer = page.locator("#layerIn");
  if ((await layer.count()) === 1) {
    const t0 = (await layer.textContent())?.trim() || "";
    await page.waitForTimeout(1600);
    const t1 = (await layer.textContent())?.trim() || "";
    if (t0 && t0 === t1) issues.push("Footer greeting (#layerIn) did not update over time");
  } else {
    issues.push("Footer greeting element (#layerIn) not found on index");
  }

  // Active nav highlights while scrolling.
  const navLinks = page.locator(".nav-links a");
  if ((await navLinks.count()) > 0) {
    const targets = ["about", "skills", "projects", "experience", "education", "references", "contact"];
    for (const id of targets) {
      const exists = (await page.locator(`#${id}`).count()) > 0;
      if (!exists) {
        issues.push(`Missing section #${id} (active-nav cannot work)`);
        continue;
      }
      await page.locator(`#${id}`).evaluate((el) => el.scrollIntoView({ block: "start", behavior: "instant" }));
      await page.waitForTimeout(250);
      const activeHref = await page.evaluate(() => document.querySelector(".nav-links a.active")?.getAttribute("href"));
      if (!activeHref || !(activeHref === `#${id}` || activeHref.endsWith(`#${id}`))) {
        issues.push(`Active nav mismatch near #${id} (active=${activeHref || "none"})`);
      }
    }
  } else {
    issues.push("Nav links not found for active-nav check");
  }

  // Hamburger (mobile widths)
  const hamburger = await checkHamburger(page, viewportWidth);
  if (hamburger.checked && !hamburger.ok) issues.push(`Mobile menu: ${hamburger.issue}`);

  return { issues };
}

async function checkProjectsBehaviors(page, viewportWidth) {
  const issues = [];

  const section = page.locator("#all-projects");
  if ((await section.count()) !== 1) issues.push("Missing #all-projects section");

  const cards = page.locator(".project-card");
  if ((await cards.count()) === 0) issues.push("No .project-card found on projects page");

  const hamburger = await checkHamburger(page, viewportWidth);
  if (hamburger.checked && !hamburger.ok) issues.push(`Mobile menu: ${hamburger.issue}`);

  return { issues };
}

async function checkContactBehaviors(page, viewportWidth) {
  const issues = [];

  const form = page.locator("#contactFormMain");
  if ((await form.count()) !== 1) issues.push("Missing contact form (#contactFormMain)");

  const hamburger = await checkHamburger(page, viewportWidth);
  if (hamburger.checked && !hamburger.ok) issues.push(`Mobile menu: ${hamburger.issue}`);

  if ((await form.count()) === 1) {
    await page.fill("#from_name", "Test User");
    await page.fill("#from_email", "test@example.com");
    await page.fill("#message", "Test message (UI-only).");
    await page.click("#submitBtn");

    // Success is shown quickly (setTimeout 300ms).
    try {
      await page.waitForSelector("#successMessage", { state: "visible", timeout: 4000 });
    } catch {
      issues.push("Contact submit did not show success UI (#successMessage) within 4s");
    }

    const errorShown = await page.evaluate(() => {
      const el = document.getElementById("errorMessage");
      if (!el) return false;
      return getComputedStyle(el).display !== "none";
    });
    if (errorShown) issues.push("Contact submit displayed error UI (#errorMessage)");
  }

  return { issues };
}

async function launchChromium() {
  // The bundled Chromium headless shell can crash in some sandboxed environments on macOS.
  // Prefer the locally installed Chrome channel if available, otherwise fall back.
  try {
    return await chromium.launch({ headless: true, channel: "chrome" });
  } catch {
    return await chromium.launch({ headless: true });
  }
}

async function runOneViewport(vp) {
  const browser = await launchChromium();
  const context = await browser.newContext({ viewport: vp });
  await addEmailJsStubRouting(context);

  const viewportReport = {
    viewport: vp,
    pages: [],
    consoleErrors: [],
    pageErrors: [],
    requestFailures: [],
    overflows: [],
    brokenAnchors: [],
    localLinkStatuses: [],
  };

  for (const p of PAGES) {
    const page = await context.newPage();
    const collectors = attachCollectors(page);

    const url = `${BASE_URL}${p.path}`;
    await page.goto(url, { waitUntil: "domcontentloaded", timeout: 20000 });
    await page.waitForTimeout(350);

    // Basic presence checks (sections load)
    if (p.name === "index.html") {
      await page.waitForSelector("#hero", { timeout: 12000 });
      await page.waitForSelector("#about", { timeout: 12000 });
      await page.waitForSelector("#references", { timeout: 12000 });
      await page.waitForSelector("footer.bcf-footer", { timeout: 12000 });
    } else if (p.name === "projects.html") {
      await page.waitForSelector("#all-projects", { timeout: 12000 });
      await page.waitForSelector("footer.bcf-footer", { timeout: 12000 });
    } else if (p.name === "contact.html") {
      await page.waitForSelector("#contactFormMain", { timeout: 12000 });
      await page.waitForSelector("footer.bcf-footer", { timeout: 12000 });
    }

    let behavior = { issues: [] };
    if (p.name === "index.html") behavior = await checkIndexBehaviors(page, vp.width);
    if (p.name === "projects.html") behavior = await checkProjectsBehaviors(page, vp.width);
    if (p.name === "contact.html") behavior = await checkContactBehaviors(page, vp.width);

    const overflow = await checkHorizontalOverflow(page);
    const anchors = await checkInPageAnchors(page);
    const localLinks = await checkLocalLinks(page);

    viewportReport.pages.push({ name: p.name, url, issues: behavior.issues });
    viewportReport.overflows.push({ page: p.name, ...overflow });
    viewportReport.brokenAnchors.push({ page: p.name, ...anchors });
    viewportReport.localLinkStatuses.push({ page: p.name, ...localLinks });

    viewportReport.consoleErrors.push(...collectors.consoleErrors.map((e) => ({ page: p.name, ...e })));
    viewportReport.pageErrors.push(...collectors.pageErrors.map((e) => ({ page: p.name, error: e })));
    viewportReport.requestFailures.push(...collectors.requestFailures.map((e) => ({ page: p.name, ...e })));

    await page.close();
  }

  await context.close();
  await browser.close();
  return viewportReport;
}

function formatViewportReport(r) {
  const lines = [];
  lines.push(`=== Viewport ${r.viewport.width}x${r.viewport.height} ===`);

  for (const p of r.pages) {
    if (!p.issues.length) lines.push(`- ${p.name}: OK`);
    else {
      lines.push(`- ${p.name}: Issues (${p.issues.length})`);
      for (const issue of p.issues) lines.push(`  - ${issue}`);
    }
  }

  const overflowPages = r.overflows.filter((o) => o.hasOverflow);
  if (!overflowPages.length) lines.push(`- Horizontal overflow: none detected`);
  else {
    for (const o of overflowPages) {
      lines.push(
        `- Horizontal overflow on ${o.page}: scrollWidth=${o.scrollWidth}, innerWidth=${o.innerWidth}${
          o.offenders?.length ? ` (offenders: ${o.offenders.map((x) => `${x.selector} ~${x.overflowPx}px`).join(", ")})` : ""
        }`
      );
    }
  }

  const broken = r.brokenAnchors
    .filter((b) => b.broken?.length)
    .map((b) => ({ page: b.page, broken: b.broken }));
  if (!broken.length) lines.push(`- Broken in-page anchors: none`);
  else broken.forEach((b) => lines.push(`- Broken in-page anchors on ${b.page}: ${b.broken.join(", ")}`));

  const placeholders = r.localLinkStatuses.flatMap((x) => x.placeholderHrefs || []);
  const placeholderHashLinks = r.brokenAnchors.flatMap((x) => x.placeholders || []);
  const placeholderTotal = uniq([...placeholders, ...placeholderHashLinks]);
  if (placeholderTotal.length) {
    lines.push(`- Placeholder links: found (e.g. ${placeholderTotal.slice(0, 4).join(", ")})`);
  }

  const badLocal = r.localLinkStatuses
    .flatMap((x) => (x.checked || []).map((y) => ({ page: x.page, ...y })))
    .filter((x) => !x.status || x.status >= 400);
  if (badLocal.length) {
    const sample = badLocal.slice(0, 6).map((x) => `${x.page}: ${x.href} -> ${x.status || "ERR"}`);
    lines.push(`- Broken local links (sample): ${sample.join(" | ")}`);
  }

  const consoleErrCount = r.consoleErrors.length;
  const pageErrCount = r.pageErrors.length;
  lines.push(`- Console errors: ${consoleErrCount}${consoleErrCount ? ` (sample: ${truncate(r.consoleErrors[0]?.text)})` : ""}`);
  lines.push(`- Page errors: ${pageErrCount}${pageErrCount ? ` (sample: ${truncate(r.pageErrors[0]?.error)})` : ""}`);

  const reqFailCount = r.requestFailures.length;
  if (!reqFailCount) lines.push(`- Failed network requests: 0`);
  else {
    const sample = r.requestFailures.slice(0, 4).map((f) => `${f.page}: ${truncate(f.url, 90)} (${f.failure || "failed"})`);
    lines.push(`- Failed network requests: ${reqFailCount} (sample: ${sample.join(" | ")})`);
  }

  return lines.join("\n");
}

async function main() {
  const reports = [];
  for (const vp of VIEWPORTS) {
    // eslint-disable-next-line no-console
    console.log(`Running checks for ${vp.width}x${vp.height}...`);
    reports.push(await runOneViewport(vp));
  }

  // eslint-disable-next-line no-console
  console.log("\n\n" + reports.map(formatViewportReport).join("\n\n"));

  // Exit non-zero if there were any issues or console/page errors.
  const totalIssues = reports.reduce((sum, r) => sum + r.pages.reduce((s, p) => s + p.issues.length, 0), 0);
  const totalConsoleErrors = reports.reduce((sum, r) => sum + r.consoleErrors.length, 0);
  const totalPageErrors = reports.reduce((sum, r) => sum + r.pageErrors.length, 0);

  if (totalIssues || totalConsoleErrors || totalPageErrors) process.exitCode = 2;
}

main().catch((e) => {
  // eslint-disable-next-line no-console
  console.error(e);
  process.exit(1);
});

