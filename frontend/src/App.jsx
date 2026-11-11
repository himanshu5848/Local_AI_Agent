import { useState } from "react";
import "./App.css";

const suggestions = [
  "Which reviews mention slow service?",
  "What do customers like most?",
  "Summarize the restaurant reviews.",
];

function App() {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  function startNewChat() {
    if (isLoading) return;
    setMessages([]);
    setMessage("");
  }

  async function handleSend(text = message) {
    const trimmedMessage = text.trim();

    if (!trimmedMessage || isLoading) return;

    const timestamp = Date.now();

    setMessages((previousMessages) => [
      ...previousMessages,
      {
        id: `${timestamp}-user`,
        text: trimmedMessage,
        sender: "user",
      },
    ]);

    setMessage("");
    setIsLoading(true);

    try {
      const API_URL = import.meta.env.VITE_API_URL;

      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: trimmedMessage,
        }),
      });

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status}`);
      }

      const data = await response.json();

      setMessages((previousMessages) => [
        ...previousMessages,
        {
          id: `${timestamp}-ai`,
          text: data.answer,
          sender: "ai",
        },
      ]);
    } catch (error) {
      console.error("Chat request failed:", error);

      setMessages((previousMessages) => [
        ...previousMessages,
        {
          id: `${timestamp}-error`,
          text: "I couldn't connect to the AI server. Check that the backend is running and try again.",
          sender: "ai",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  }

  return (
    <main className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">AI</div>
          <div className="brand-copy">
            <h1>ReviewMind AI</h1>
            <p>AI-Powered Restaurant Review Intelligence</p>
          </div>
        </div>

        <div className="local-status">
          <span className="status-dot" />
          Local
        </div>
      </header>

      <div className="workspace">
        <aside className="sidebar">
          <button
            className="new-chat-button"
            type="button"
            onClick={startNewChat}
            disabled={isLoading}
          >
            <span>＋</span> New chat
          </button>

          <div className="sidebar-section">
            <p className="sidebar-heading">WORKSPACE</p>

            <div className="sidebar-item active">
              <span>◈</span> Chat assistant
            </div>

            <div className="sidebar-item">
              <span>⌕</span> Review search
            </div>

            <div className="sidebar-item">
              <span>▤</span> Knowledge base
            </div>
          </div>

          <div className="sidebar-info">
            <span className="sidebar-info-icon">✳</span>
            <p>Explore restaurant reviews with AI-powered insights.</p>
          </div>
        </aside>

        <section className="chat-panel">
          <div className="chat-area">
            {messages.length === 0 ? (
              <div className="welcome">
                <div className="welcome-icon">✳</div>

                <h2>What can I help with?</h2>

                <p>Ask questions and explore your restaurant reviews.</p>

                <div className="suggestions">
                  {suggestions.map((suggestion) => (
                    <button
                      className="suggestion"
                      key={suggestion}
                      type="button"
                      onClick={() => handleSend(suggestion)}
                      disabled={isLoading}
                    >
                      {suggestion}
                      <span>↗</span>
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="messages">
                {messages.map((item) => (
                  <div
                    className={`message-row ${item.sender}-row`}
                    key={item.id}
                  >
                    {item.sender === "ai" && (
                      <div className="ai-avatar">AI</div>
                    )}

                    <div className={`message-bubble ${item.sender}-message`}>
                      {item.text}
                    </div>
                  </div>
                ))}

                {isLoading && (
                  <div className="message-row ai-row">
                    <div className="ai-avatar">AI</div>
                    <div className="message-bubble ai-message loading">
                      <span />
                      <span />
                      <span />
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          <footer className="composer-area">
            <div className="composer">
              <textarea
                rows={1}
                placeholder="Ask anything about your reviews..."
                aria-label="Your message"
                value={message}
                disabled={isLoading}
                onChange={(event) => setMessage(event.target.value)}
                onKeyDown={handleKeyDown}
              />

              <button
                className="send-button"
                type="button"
                aria-label="Send message"
                onClick={() => handleSend()}
                disabled={!message.trim() || isLoading}
              >
                ↑
              </button>
            </div>

            <p className="footer-note">
              Powered by your local AI model · Enter to send
            </p>
          </footer>
        </section>
      </div>
    </main>
  );
}

export default App;
