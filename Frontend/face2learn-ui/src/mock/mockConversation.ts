export type Message =  {
  role: "user" | "assistant"
  text?: string
  mediaUrl?: string
}

export const mockAssistantResponse: Message = {
  role: "assistant",
  text: `O mașină Turing este un model teoretic de calcul care descrie modul în care un algoritm procesează informația pas cu pas. Practic, o poți vedea ca pe un calculator extrem de simplu, dar suficient de puternic pentru a simula orice program.`,
  mediaUrl: "/avatar_lipsync.mp4"
}
