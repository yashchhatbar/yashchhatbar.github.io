/* ================================
Knowledge JSON Files
================================ */

const KNOWLEDGE_FILES = {
  projects: "knowledge/projects.json",
  skills: "knowledge/skills.json",
  experience: "knowledge/experience.json",
  about: "knowledge/about.json",
  contact: "knowledge/contact.json",
};

let knowledgeCache = {};



/* ================================
Scroll chat automatically
================================ */

const scrollChat = () => {

  const container = document.getElementById("ai-messages");
  if (!container) return;

  requestAnimationFrame(() => {
    container.scrollTop = container.scrollHeight;
  });

};


/* ================================
Clean AI Output
================================ */

const cleanAIResponse = (text) => {

  if (!text) return "";

  let cleaned = text
    .replace(/\*\*/g, "")
    .replace(/\*/g, "")
    .replace(/`/g, "")
    .replace(/#{1,6}/g, "")
    .replace(/^\d+\.\s*/gm, "")
    .replace(/^\-\s*/gm, "")
    .replace(/^\•\s*/gm, "")
    .replace(/\n{2,}/g, "\n")
    .trim();

  const sentences = cleaned.split(/(?<=[.!?])\s+/);
  cleaned = sentences.slice(0, 2).join(" ");

  return cleaned.replace(/\n/g, "<br>");

};


/* ================================
Random Project Selector
================================ */

const getRandomProjects = () => {

  const projects = knowledgeCache.projects?.projects || [];

  if (projects.length <= 2) return projects;

  const shuffled = [...projects];

  for (let i = shuffled.length - 1; i > 0; i--) {

    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];

  }

  return shuffled.slice(0, 2);

};


/* ================================
Render Projects
================================ */

const renderProjects = () => {

  const projects = getRandomProjects();

  let html = "Here are two of Yash's projects:<br><br>";

  projects.forEach(p => {

    html += `
<b>${p.name}</b><br>
Tech: ${(p.tech || []).join(", ")}<br>
Impact: ${p.impact || ""}<br><br>
`;

  });

  return html;

};


/* ================================
Render Skills
================================ */

const renderSkills = () => {

  const skills = knowledgeCache.skills?.skills || {};

  let html = "Yash's key skills:<br><br>";

  Object.keys(skills).forEach(category => {

    html += `<b>${category}</b>: ${skills[category].slice(0, 4).join(", ")}<br>`;

  });

  return html;

};


/* ================================
Render Experience
================================ */

const renderExperience = () => {

  const exp = knowledgeCache.experience?.experience || [];

  let html = "Professional experience:<br><br>";

  exp.forEach(e => {

    html += `<b>${e.role}</b> — ${e.company}<br>${e.period}<br><br>`;

  });

  return html;

};


/* ================================
Render About
================================ */

const renderAbout = () => {

  const a = knowledgeCache.about?.about || {};

  return `
<b>${a.name}</b><br>
${a.title}<br>
Focus: ${(a.focus || []).join(", ")}
`;

};


/* ================================
Render Contact
================================ */

const renderContact = () => {

  const c = knowledgeCache.contact?.contact || {};

  return `
Contact Yash:<br><br>
Email: ${c.email}<br>
LinkedIn: ${c.linkedin}<br>
GitHub: ${c.github}
`;

};


/* ================================
Load Knowledge
================================ */

const loadKnowledge = async () => {

  for (const key in KNOWLEDGE_FILES) {

    try {

      const res = await fetch(KNOWLEDGE_FILES[key]);
      knowledgeCache[key] = res.ok ? await res.json() : {};

    } catch {

      knowledgeCache[key] = {};

    }



  };


  /* ================================
  Inject Assistant UI
  ================================ */

  const injectAssistantUI = () => {

    const wrapper = document.createElement("div");
    wrapper.id = "ai-assistant-wrapper";

    wrapper.innerHTML = `
<div id="ai-assistant-window">

<div class="assistant-header">
<i class="fas fa-robot"></i> AI Portfolio Assistant
<button class="assistant-close" id="ai-close">
<i class="fas fa-times"></i>
</button>
</div>

<div class="assistant-messages" id="ai-messages"></div>

<div class="assistant-input-area">
<input id="ai-input" placeholder="Ask about projects, skills..." autocomplete="off"/>
<button id="ai-send">Send</button>
</div>

</div>

<button id="ai-assistant-toggle">
<i class="icon-chat fas fa-comment-dots"></i>
<i class="icon-close fas fa-chevron-down"></i>
</button>
`;

    document.body.appendChild(wrapper);

    setupEvents();

  };


  /* ================================
  Welcome Message
  ================================ */

  const sendWelcomeMessage = () => {

    const container = document.getElementById("ai-messages");

    container.insertAdjacentHTML("beforeend", `

<div class="chat-bubble ai">

👋 Hi! I'm Yash's AI assistant.<br><br>

Who are you?

<div class="assistant-suggestions">

<button class="suggestion-btn recruiter" onclick="setMode('recruiter')">👔 Recruiter</button>
<button class="suggestion-btn developer" onclick="setMode('developer')">💻 Developer</button>
<button class="suggestion-btn student" onclick="setMode('student')">🎓 Student</button>

</div>

</div>

`);

  };


  /* ================================
  Visitor Mode
  ================================ */



  const container = document.getElementById("ai-messages");

  container.insertAdjacentHTML("beforeend", `

<div class="chat-bubble ai">

Great! Ask about:

<div class="assistant-suggestions">

<button class="suggestion-btn" onclick="askSuggestion('Show me your projects')">Projects</button>
<button class="suggestion-btn" onclick="askSuggestion('What skills do you have')">Skills</button>
<button class="suggestion-btn" onclick="askSuggestion('Explain your experience')">Experience</button>
<button class="suggestion-btn" onclick="askSuggestion('Tell me about yourself')">About</button>
<button class="suggestion-btn" onclick="askSuggestion('How can I contact you')">Contact</button>

</div>

</div>

`);

  scrollChat();

};


/* ================================
Suggestion Helper
================================ */

window.askSuggestion = (q) => {

  const input = document.getElementById("ai-input");
  input.value = q;
  document.getElementById("ai-send").click();

};


/* ================================
Events
================================ */

const setupEvents = () => {

  const toggleBtn = document.getElementById("ai-assistant-toggle");
  const closeBtn = document.getElementById("ai-close");
  const sendBtn = document.getElementById("ai-send");
  const input = document.getElementById("ai-input");
  const wrapper = document.getElementById("ai-assistant-wrapper");

  toggleBtn.addEventListener("click", () => {

    wrapper.classList.toggle("open");



  });

  closeBtn.addEventListener("click", () => {

    wrapper.classList.remove("open");

  });

  sendBtn.addEventListener("click", () => {

    const value = input.value.trim();
    if (!value) return;

    processQuestion(value);
    input.value = "";

  });

  input.addEventListener("keypress", (e) => {

    if (e.key === "Enter") {

      const value = input.value.trim();
      if (!value) return;

      processQuestion(value);
      input.value = "";

    }

  });

  sendWelcomeMessage();

};





/* ================================
Process Question
================================ */

const processQuestion = async (question) => {

  const q = question.toLowerCase();
  const container = document.getElementById("ai-messages");

  container.insertAdjacentHTML(
    "beforeend",
    `<div class="chat-bubble user">${question}</div>`
  );

  scrollChat();

  const bubble = document.createElement("div");
  bubble.className = "chat-bubble ai";
  container.appendChild(bubble);


  /* ===== Instant responses for mobile ===== */

  if (q.includes("hello") || q.includes("hi")) {
    bubble.innerHTML = "Hi 👋 I'm Yash's AI assistant. Ask me about projects, skills, experience or contact.";
    return;
  }

  if (q.includes("who are you") || q.includes("what do you do")) {
    bubble.innerHTML = "I help visitors explore Yash Chhatbar's AI portfolio including projects, skills, and experience.";
    return;
  }

  if (q.includes("study") || q.includes("education")) {
    bubble.innerHTML = "Yash studies computer science and focuses on Artificial Intelligence, Machine Learning, and Data Science.";
    return;
  }


  /* ===== Portfolio answers ===== */

  if (q.includes("project")) {
    bubble.innerHTML = renderProjects();
    return;
  }

  if (q.includes("skill")) {
    bubble.innerHTML = renderSkills();
    return;
  }

  if (q.includes("experience")) {
    bubble.innerHTML = renderExperience();
    return;
  }

  if (q.includes("about")) {
    bubble.innerHTML = renderAbout();
    return;
  }

  if (q.includes("contact")) {
    bubble.innerHTML = renderContact();
    return;
  }


  /* ===== AI fallback ===== */

  bubble.innerHTML = `<div class="typing"><span></span><span></span><span></span></div>`;

  try {
    const response = await fetch("http://localhost:8000/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: question })
    });

    if (!response.ok) throw new Error("Backend error");

    const data = await response.json();
    bubble.innerHTML = cleanAIResponse(data.answer);
  } catch (error) {
    bubble.innerHTML = "AI assistant is temporarily unavailable.";
  }

  scrollChat();

};


/* ================================
Initialize
================================ */

document.addEventListener("DOMContentLoaded", async () => {

  await loadKnowledge();
  injectAssistantUI();

});
