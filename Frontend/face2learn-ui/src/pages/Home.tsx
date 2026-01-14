import { useState, useEffect, useRef } from "react"
import "../chat.css"
import { postChat, getVideoModel, setVideoModel, type VideoModelType } from "../api/chatApi"

type Message = {
  role: "user" | "assistant"
  text: string
  mediaUrl?: string
  sources?: string[]
}

// --- COMPONENTĂ PENTRU EFECTUL DE TYPEWRITER ---
function AiMessage({ text, mediaUrl, sources }: { text: string, mediaUrl?: string, sources?: string[] }) {
  const [displayedText, setDisplayedText] = useState("")
  const [isTyping, setIsTyping] = useState(true)

  useEffect(() => {
    let index = 0
    // Viteza de scriere (mai mic = mai rapid)
    const speed = 20 

    const interval = setInterval(() => {
      if (index < text.length) {
        // Adăugăm câte un caracter sau bucăți mici pentru fluiditate
        setDisplayedText((prev) => prev + text.charAt(index))
        index++
      } else {
        clearInterval(interval)
        setIsTyping(false)
      }
    }, speed)

    return () => clearInterval(interval)
  }, [text])

  return (
    <div className="bubble ai">
      {/* Textul generat */}
      <p>{displayedText}</p>

      {/* Video și Surse apar DOAR după ce textul s-a terminat de scris */}
      {!isTyping && (
        <div className="fade-in-content">
          {mediaUrl && (
            <div className="video-container">
              <video src={mediaUrl} autoPlay controls />
            </div>
          )}

          {sources && sources.length > 0 && (
            <div className="sources">
              <strong>📚 Surse utilizate:</strong>
              <ul>
                {sources.map((s, idx) => (
                  <li key={idx}>{s}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
// --------------------------------------------------

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [selectedModel, setSelectedModel] = useState<VideoModelType>("wav2lip")

  useEffect(() => {
    getVideoModel()
      .then((data) => setSelectedModel(data.current_model))
      .catch((err) => console.error("Eroare la preluarea modelului:", err))
  }, [])

  // Auto-scroll la ultimul mesaj
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, loading])

  const handleModelChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newModel = e.target.value as VideoModelType
    setSelectedModel(newModel)
    try {
      await setVideoModel(newModel)
    } catch (error) {
      console.error("Nu s-a putut schimba modelul:", error)
    }
  }

  async function handleSend(text: string) {
    if (!text.trim()) return
    
    // Adaugă mesaj user
    setMessages((prev) => [...prev, { role: "user", text }])
    setInput("")
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
          text: "Îmi pare rău, a apărut o eroare de conexiune. Te rog să încerci din nou.",
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page">
      <div className="app">
        <div className="header">
           ✨ Face2Learn AI
        </div>

        <div className="messages">
          {/* WELCOME SCREEN DACĂ NU SUNT MESAJE */}
          {messages.length === 0 && (
            <div className="welcome-screen">
              <div className="welcome-icon">🎓</div>
              <div className="welcome-text">
                <h2>Salut! Sunt asistentul tău digital.</h2>
                <p>Întreabă-mă orice despre cursurile tale, iar eu îți voi răspunde text și video.</p>
              </div>
            </div>
          )}

          {messages.map((m, i) => (
            m.role === "user" ? (
              // MESAJ USER (Simplu)
              <div key={i} className="bubble user">
                <p>{m.text}</p>
              </div>
            ) : (
              // MESAJ AI (Cu efect Typewriter)
              <AiMessage 
                key={i} 
                text={m.text} 
                mediaUrl={m.mediaUrl} 
                sources={m.sources} 
              />
            )
          ))}

          {/* INDICATOR DE ÎNCĂRCARE (Cât timp serverul procesează) */}
          {loading && (
            <div className="bubble ai">
              <div className="typing-indicator">
                <div className="dot"></div>
                <div className="dot"></div>
                <div className="dot"></div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        <div className="input-bar">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={loading ? "Procesez răspunsul..." : "Scrie întrebarea ta aici..."}
            disabled={loading}
            onKeyDown={(e) => e.key === "Enter" && !loading && handleSend(input)}
          />

          <select 
            className="model-select"
            value={selectedModel}
            onChange={handleModelChange}
            disabled={loading}
          >
            <option value="sadtalker">SadTalker</option>
            <option value="wav2lip">Wav2Lip</option>
          </select>

          <button onClick={() => handleSend(input)} disabled={loading || !input.trim()}>
            {loading ? "..." : "➤"}
          </button>
        </div>
      </div>
    </div>
  )
}