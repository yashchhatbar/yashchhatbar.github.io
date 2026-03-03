// This module acts as a bridge to the Python FastAPI backend
// so the heavy LLM execution happens on the server rather than the browser.

let visitorMode = "general";

export const setVisitorMode = (mode) => {
  visitorMode = mode;
};

export const initializeContext = (contextText) => {
  // Context mapping is handled by the backend RAG service
};

export const loadModel = async (progressCallback = () => { }) => {
  // The backend model is loaded on the server
  // This just ensures the UI proceeds without delay
  progressCallback({
    text: "Backend AI model is active."
  });
};

export const generateResponse = async (message, streamCallback) => {
  try {
    const pageContext = window.location.pathname || "/";
    const userMessage = visitorMode !== "general" ? `${message} (I am a ${visitorMode})` : message;

    const response = await fetch("http://localhost:8000/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: userMessage,
        page: pageContext
      })
    });

    if (!response.ok) {
      throw new Error("Server error");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let fullText = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      fullText += chunk;
      streamCallback(fullText);
    }

  } catch (error) {
    streamCallback("I cannot connect to the local AI model. Please ensure Ollama is running.");
  }
};
