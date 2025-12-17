export type ChatResponse = {
  text: string
  video_url: string | null
  sources: string[]
  metrics: {
    search: number
    llm: number
    video: number
    total: number
  }
}

export async function postChat(message: string): Promise<ChatResponse> {
  const res = await fetch("http://localhost:8000/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  })

  // dacă backendul e "Sistemul se încarcă..." (500) sau alt status
  if (!res.ok) {
    let detail = ""
    try {
      const err = await res.json()
      detail = err?.detail ? ` (${err.detail})` : ""
    } catch {
      // ignore
    }
    throw new Error(`Backend error: ${res.status}${detail}`)
  }

  return res.json()
}
