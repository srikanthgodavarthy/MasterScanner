// live.js — SSE client for §9 real-time update strategy.
// Connects to /stream/prices and patches the specific price cell / arc
// element for the ticker in the event, rather than re-fetching HTML.
// Falls back silently if EventSource is unsupported or the connection
// drops (browser will auto-retry SSE by spec; no polling fallback is
// wired here yet — see migration-requirements §9 "Fallback" note).
function initLiveTicks(streamUrl) {
  if (!window.EventSource) return;

  const source = new EventSource(streamUrl);

  source.addEventListener('tick', (evt) => {
    let data;
    try {
      data = JSON.parse(evt.data);
    } catch (e) {
      return;
    }
    const { ticker, price, conviction } = data;

    const priceCell = document.querySelector(`.price-cell[data-ticker="${ticker}"]`);
    if (priceCell) {
      priceCell.textContent = `₹${price.toFixed(2)}`;
      priceCell.classList.add('tick-flash');
      setTimeout(() => priceCell.classList.remove('tick-flash'), 600);
    }

    const arc = document.querySelector(`.conviction-arc[data-ticker="${ticker}"]`);
    if (arc && window.updateConvictionArc) {
      window.updateConvictionArc(arc, conviction);
    }
  });

  source.onerror = () => {
    // EventSource auto-reconnects; nothing to do here beyond letting it retry.
  };
}
