import { type Message } from "../mock/mockConversation"

export default function ChatBubble({ message }: { message: Message }) {
  return (
    <div className={`bubble ${message.role === "user" ? "user" : "ai"}`}>
      {message.text && <p>{message.text}</p>}

      {message.mediaUrl && (
        <video src={message.mediaUrl} controls />
      )}
    </div>
  )
}
