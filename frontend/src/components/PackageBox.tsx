// Ready-to-post text for a clip: YouTube title, thumbnail text, description, chapters.
import { useState } from "react";
import { Check, Copy, Youtube } from "lucide-react";
import type { Clip } from "../api";
import { fmtTime } from "../lib/util";

export function PackageBox({ c }: { c: Clip }) {
  const pk = c.package;
  const [copied, setCopied] = useState("");
  if (!pk || !(pk.yt_title || pk.thumbnail_text || pk.description)) return null;

  const chapters = (pk.chapters ?? []).map(([t, label]) => `${fmtTime(t)} ${label}`).join("\n");
  const copy = async (key: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(key);
      setTimeout(() => setCopied(""), 1300);
    } catch {
      /* clipboard blocked */
    }
  };
  const Btn = ({ k, text }: { k: string; text: string }) => (
    <button className="icon-btn" title="Copy" onClick={() => copy(k, text)}>
      {copied === k ? <Check size={13} /> : <Copy size={13} />}
    </button>
  );

  return (
    <div className="package">
      <div className="package-head"><Youtube size={14} /> Ready to post</div>
      {pk.yt_title && (
        <div className="package-row">
          <span className="package-label">Title</span>
          <span className="grow">{pk.yt_title}</span>
          <Btn k="t" text={pk.yt_title} />
        </div>
      )}
      {pk.thumbnail_text && (
        <div className="package-row">
          <span className="package-label">Thumbnail</span>
          <span className="grow thumb-text">{pk.thumbnail_text}</span>
          <Btn k="th" text={pk.thumbnail_text} />
        </div>
      )}
      {pk.description && (
        <div className="package-row">
          <span className="package-label">Description</span>
          <span className="grow muted small">{pk.description}</span>
          <Btn k="d" text={`${pk.description}${chapters ? `\n\n${chapters}` : ""}\n\n${c.hashtags.join(" ")}`} />
        </div>
      )}
      {chapters && (
        <div className="package-row">
          <span className="package-label">Chapters</span>
          <pre className="grow chapters">{chapters}</pre>
          <Btn k="c" text={chapters} />
        </div>
      )}
    </div>
  );
}
