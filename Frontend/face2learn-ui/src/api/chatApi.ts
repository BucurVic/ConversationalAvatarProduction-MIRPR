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

// --- TIPURI NOI PENTRU SETĂRI ---
export type VideoModelType = "sadtalker" | "wav2lip"

export type VideoModelResponse = {
  current_model: VideoModelType
}

export type SetVideoModelResponse = {
  status: string
  current_model: VideoModelType
  message: string
}
// -------------------------------

const API_BASE = "http://localhost:8000"

export async function postChat(message: string): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  })

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

// --- FUNCȚII NOI PENTRU SCHIMBAREA MODELULUI ---

/**
 * GET: Află care este modelul curent (sadtalker sau wav2lip)
 */
export async function getVideoModel(): Promise<VideoModelResponse> {
  const res = await fetch(`${API_BASE}/settings/video-model`, {
    method: "GET",
  })

  if (!res.ok) {
    throw new Error(`Failed to get video model: ${res.status}`)
  }

  return res.json()
}

/**
 * POST: Schimbă modelul de generare video
 */
export async function setVideoModel(modelName: VideoModelType): Promise<SetVideoModelResponse> {
  const res = await fetch(`${API_BASE}/settings/video-model`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_name: modelName }),
  })

  if (!res.ok) {
    let detail = ""
    try {
      const err = await res.json()
      detail = err?.detail ? ` (${err.detail})` : ""
    } catch {
      // ignore
    }
    throw new Error(`Failed to set video model: ${res.status}${detail}`)
  }

  return res.json()
}