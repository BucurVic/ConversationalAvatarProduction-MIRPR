export default function AvatarVideo({ url }: { url: string }) {
  return (
    <div className="mt-4">
      <video
        src={url}
        controls
        className="w-full rounded-xl border border-slate-700"
      />
    </div>
  )
}
