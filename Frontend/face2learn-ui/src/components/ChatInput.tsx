import { useState } from "react"

export default function ChatInput({
  onSend,
}: {
  onSend: (text: string) => void
}) {
  const [value, setValue] = useState("")

  function send() {
    if (!value.trim()) return
    onSend(value)
    setValue("")
  }

  return (
    <div className="chat-input">
      <input
        placeholder="Scrie întrebarea ta…"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && send()}
      />
      <button onClick={send}>Trimite</button>
    </div>
  )
}
