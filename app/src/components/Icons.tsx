import type { SVGProps } from "react";

// Tactical icon set: 24px grid, 1.6 stroke, mitred joins, chamfered corners.
const PATHS = {
  mic: "M9 3 H15 V12 L12 15 L9 12 Z M5 10 V12 L12 18 L19 12 V10 M12 18 V21 M8 21 H16",
  core: "M7 7 H17 V17 H7 Z M10 10 H14 V14 H10 Z M9 3 V7 M12 3 V7 M15 3 V7 M9 17 V21 M12 17 V21 M15 17 V21 M3 9 H7 M3 15 H7 M17 9 H21 M17 15 H21",
  speaker: "M3 9 H7 L12 5 V19 L7 15 H3 Z M15.5 9 L17.5 12 L15.5 15 M18.5 6 L21.5 12 L18.5 18",
  target: "M12 2 V7 M12 17 V22 M2 12 H7 M17 12 H22 M12 8 L16 12 L12 16 L8 12 Z",
  timer: "M9 2 H15 M12 2 V5 M8 5 H16 L20 9 V15 L16 19 H8 L4 15 V9 Z M12 9 V12 L14.5 14",
  gauge: "M3 17 A9 9 0 1 1 21 17 M12 17 L16.5 9.5 M3 17 H6 M18 17 H21",
  chip: "M4 6 H20 V18 H4 Z M8 3 V6 M12 3 V6 M16 3 V6 M8 18 V21 M12 18 V21 M16 18 V21 M8 10 H12 V14 H8 Z M14 10 H16 M14 14 H16",
  pitch: "M2 13 H5 L7 8 L10 18 L13 4 L16 16 L18 11 H22",
  index: "M4 4 H10 V10 H4 Z M14 4 H20 V10 H14 Z M4 14 H10 V20 H4 Z M14 14 H20 V20 H14 Z M14 14 L20 20",
  shield: "M12 3 L20 6 V13 L12 21 L4 13 V6 Z M9 12 L11 14 L15 10",
  blocks: "M3 7 H7 V17 H3 Z M10 4 H14 V20 H10 Z M17 9 H21 V15 H17 Z",
  power: "M12 3 V11 M7 5.5 L4 9 V15 L9 20 H15 L20 15 V9 L17 5.5",
  plus: "M12 5 V19 M5 12 H19",
  wave: "M2 12 H4 M6 8 V16 M9 5 V19 M12 9 V15 M15 6 V18 M18 9 V15 M21 11 V13",
} as const;

export type IconName = keyof typeof PATHS;

export function Icon({ name, size = 16, ...rest }: { name: IconName; size?: number } & SVGProps<SVGSVGElement>) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinejoin="miter"
      strokeLinecap="square"
      {...rest}
    >
      <path d={PATHS[name]} />
    </svg>
  );
}
