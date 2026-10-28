import { useId } from "react";

// Split mask: a cut-corner visor halved by a voice waveform, cyan (you) on the left,
// accent (the voice you become) on the right, with angled eye slits.
export const MASK_PATH = "M30 12 H98 L116 32 V70 L84 116 H44 L12 70 V32 Z";
// the seam stays between x 59..69 across the eye band so it never touches the slits
export const SEAM = "M66 0 L60 22 L68 36 L61 50 L67 62 L58 80 L68 98 L61 128";
export const SPLIT_LEFT = `${SEAM} H0 V0 Z`;
export const SPLIT_RIGHT = `${SEAM} H128 V0 Z`;
export const EYE_LEFT = "M30 50 L54 45 L57 55 L33 60 Z";
export const EYE_RIGHT = "M98 50 L74 45 L71 55 L95 60 Z";

export function LogoMark({ size = 22, left = "#4fd8ff", right = "var(--accent)" }: { size?: number; left?: string; right?: string }) {
  const id = useId().replace(/:/g, "");
  return (
    <svg width={size} height={size} viewBox="0 0 128 128" className="shrink-0 overflow-visible">
      <defs>
        <clipPath id={`m${id}`}>
          <path d={MASK_PATH} />
        </clipPath>
        <linearGradient id={`l${id}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor={left} />
          <stop offset="1" stopColor={left} stopOpacity="0.55" />
        </linearGradient>
        <linearGradient id={`r${id}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor={right} />
          <stop offset="1" stopColor={right} stopOpacity="0.6" />
        </linearGradient>
      </defs>
      <g clipPath={`url(#m${id})`}>
        <path d={SPLIT_LEFT} fill={`url(#l${id})`} />
        <path d={SPLIT_RIGHT} fill={`url(#r${id})`} />
        <path d={SEAM} fill="none" stroke="#06070a" strokeWidth="4" />
        <path d={EYE_LEFT} fill="#06070a" />
        <path d={EYE_RIGHT} fill="#06070a" />
      </g>
      <path d={MASK_PATH} fill="none" stroke="rgb(255 255 255 / 0.35)" strokeWidth="2" />
    </svg>
  );
}

// Hand-built angular letterforms on a 24-unit cap height.
const GLYPHS: Record<string, { d: string; w: number }> = {
  M: { d: "M0 24 V3 L3 0 H7 L12 9 L17 0 H21 L24 3 V24 H19 V10 L13.5 19 H10.5 L5 10 V24 Z", w: 24 },
  I: { d: "M0 3 L3 0 H6 V24 H0 Z", w: 6 },
  C: { d: "M21 0 H6 L0 6 V18 L6 24 H21 V19 H8 L5 16 V8 L8 5 H21 Z", w: 21 },
};

/** "MIMIC" with a glitch slice: the band above the cut is nudged sideways. */
export function Wordmark({ height = 14, color = "var(--color-text)" }: { height?: number; color?: string }) {
  const id = useId().replace(/:/g, "");
  const gap = 4;
  let x = 0;
  const parts = "MIMIC".split("").map((ch) => {
    const g = GLYPHS[ch];
    const el = <path key={x} d={g.d} transform={`translate(${x} 0)`} />;
    x += g.w + gap;
    return el;
  });
  const width = x - gap;
  return (
    <svg height={height} viewBox={`-2 -1 ${width + 4} 26`} className="shrink-0 overflow-visible" style={{ width: (height * (width + 4)) / 26 }}>
      <defs>
        <clipPath id={`t${id}`}>
          <path d={`M-4 -2 H${width + 6} V8 L-4 13 Z`} />
        </clipPath>
        <clipPath id={`b${id}`}>
          <path d={`M-4 13 L${width + 6} 8 V27 H-4 Z`} />
        </clipPath>
      </defs>
      <g fill={color} clipPath={`url(#b${id})`}>{parts}</g>
      <g fill={color} clipPath={`url(#t${id})`} transform="translate(1.6 0)">{parts}</g>
      <path d={`M-2 13.4 L${width + 2} 8.4`} stroke="var(--accent)" strokeWidth="0.8" opacity="0.9" />
    </svg>
  );
}
