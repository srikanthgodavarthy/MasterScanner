// tailwind.config.js — proposed production config. The CDN <script> in
// templates/base.html is a dev-speed shortcut; before launch, run the
// Tailwind CLI against this config and swap base.html to link
// static/dist/output.css instead (per migration-requirements §7).
//
// Colors are duplicated from base.html's inline config. Once
// design-tokens.json is available, generate BOTH from it with a single
// small script instead of hand-keeping these in sync.
const fs = require('fs');
const path = require('path');

let tokens = {};
try {
  tokens = JSON.parse(fs.readFileSync(path.join(__dirname, 'design-tokens.json'), 'utf8'));
} catch (e) {
  // design-tokens.json not present in this migration deliverable yet —
  // fall back to the placeholder palette also used in base.html.
  tokens = {
    colors: {
      'bg-0': '#0B0E14', 'bg-1': '#11161F', 'bg-2': '#181F2B',
      'border-0': '#232B3A', 'text-0': '#E6EDF3', 'text-1': '#8B98A9',
      'interactive': '#3B82F6', 'elite': '#F5B700', 'actionable': '#22C55E',
      'watch': '#38BDF8', 'monitor': '#64748B', 'danger': '#EF4444',
    },
  };
}

module.exports = {
  darkMode: 'class',
  content: ['./templates/**/*.html', './static/js/**/*.js'],
  theme: {
    extend: {
      colors: tokens.colors,
      fontFamily: {
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
        sans: ['Inter', 'ui-sans-serif', 'system-ui'],
      },
    },
  },
  plugins: [],
};
