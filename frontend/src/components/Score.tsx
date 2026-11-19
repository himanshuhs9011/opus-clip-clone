import type { Score } from "../api";
import { gradeOf, scoreTone } from "../lib/util";

export const SUBSCORES: [keyof Score, string, string][] = [
  ["hook", "Hook", "Does the first 3 seconds stop the scroll"],
  ["retention", "Flow", "Pacing, no rambling, dead air or filler"],
  ["value", "Value", "Insight, numbers, how-to, contrast"],
  ["emotion", "Emotion", "Energy, stakes, vocal dynamics"],
  ["quotability", "Trend", "Shareable one-liners, loopable ending"],
];

export function ScoreRing({ value, size = 64 }: { value: number; size?: number }) {
  const r = (size - 8) / 2;
  const c = 2 * Math.PI * r;
  const tone = scoreTone(value);
  return (
    <div className={`ring tone-${tone}`} style={{ width: size, height: size }}>
      <svg width={size} height={size}>
        <circle cx={size / 2} cy={size / 2} r={r} className="ring-bg" />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          className="ring-fg"
          strokeDasharray={c}
          strokeDashoffset={c * (1 - value / 100)}
        />
      </svg>
      <div className="ring-label">
        <b style={{ fontSize: size * 0.32 }}>{value}</b>
      </div>
    </div>
  );
}

export function ScoreBadge({ score }: { score: Score | null }) {
  if (!score) return null;
  return (
    <div className={`score-badge tone-${scoreTone(score.total)}`}>
      <b>{score.total}</b>
      <span>/100</span>
      <i>{score.grade || gradeOf(score.total)}</i>
    </div>
  );
}

export function SubScores({ score, compact = false }: { score: Score; compact?: boolean }) {
  return (
    <div className={`subscores ${compact ? "compact" : ""}`}>
      {SUBSCORES.map(([key, label, hint]) => {
        const v = Number(score[key]) || 0;
        return (
          <div className="sub" key={key} title={hint}>
            <span className="sub-label">{label}</span>
            <div className="sub-track">
              <div className={`sub-fill tone-${scoreTone(v)}`} style={{ width: `${v}%` }} />
            </div>
            <span className="sub-grade">{compact ? v : gradeOf(v)}</span>
          </div>
        );
      })}
    </div>
  );
}
