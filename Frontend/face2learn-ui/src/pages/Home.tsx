import { useState } from "react"
import "../chat.css"
import { postChat } from "../api/chatApi"

type Message = {
  role: "user" | "assistant"
  text: string
  mediaUrl?: string
  sources?: string[]
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)

  async function handleSend(text: string) {
    setMessages((prev) => [...prev, { role: "user", text }])
    setLoading(true)

    try {
      const data = await postChat(text)

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: data.text,
          mediaUrl: data.video_url || undefined,
          sources: data.sources,
        },
      ])
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: "A apărut o eroare la server. Încearcă din nou.",
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  function send() {
    if (!input.trim()) return
    handleSend(input)
    setInput("")
  }

  return (
    <div className="page">
      <div className="app">
        <div className="header">💬 Face2Learn</div>

        <div className="messages">
          {messages.map((m, i) => (
           <div
            key={i}
            className={`bubble ${m.role === "user" ? "user" : "ai"}`}
            >
            <p>{m.text}</p>

            {m.mediaUrl && <video src={m.mediaUrl} controls />}

            {m.role === "assistant" && m.sources && m.sources.length > 0 && (
                <div className="sources">
                📚 Surse:
                <ul>
                    {m.sources.map((s, idx) => (
                    <li key={idx}>{s}</li>
                    ))}
                </ul>
                </div>
            )}
            </div>

          ))}

          {loading && (
            <div className="bubble ai">Asistentul scrie…</div>
          )}
        </div>

        <div className="input-bar">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={loading ? "Asistentul răspunde…" : "Scrie întrebarea ta…"}
            disabled={loading}
            onKeyDown={(e) => e.key === "Enter" && !loading && send()}
            />

          <button onClick={send} disabled={loading}>
            {loading ? "…" : "Trimite"}
            </button>
        </div>
      </div>
    </div>
  )
}
