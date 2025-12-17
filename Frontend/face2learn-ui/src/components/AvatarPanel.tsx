export default function AvatarPanel({ videoUrl }: { videoUrl?: string }) {
  return (
    <div className="bg-slate-900 rounded-2xl p-4 flex flex-col items-center justify-center">
      {videoUrl ? (
        <video
          src={videoUrl}
          controls
          className="rounded-xl w-full"
        />
      ) : (
        <img
          src="/avatar_idle.png"
          className="w-full rounded-xl opacity-80"
        />
      )}
    </div>
  )
}
