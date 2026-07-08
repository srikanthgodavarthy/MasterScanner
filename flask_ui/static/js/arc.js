// arc.js — updates a server-rendered conviction arc SVG in place when a
// live tick changes the score, without a full HTMX swap.
function updateConvictionArc(svgEl, score) {
  const circle = svgEl.querySelector('.conviction-arc-fill');
  const text = svgEl.querySelector('text');
  if (!circle || !text) return;

  const r = parseFloat(circle.getAttribute('r'));
  const circumference = 2 * Math.PI * r;
  const offset = circumference * (1 - score / 100);
  circle.style.transition = 'stroke-dashoffset 400ms ease';
  circle.setAttribute('stroke-dashoffset', offset);
  text.textContent = score;

  const color = score >= 85 ? '#F5B700' : score >= 70 ? '#22C55E' : score >= 55 ? '#38BDF8' : '#64748B';
  circle.setAttribute('stroke', color);
}
window.updateConvictionArc = updateConvictionArc;
