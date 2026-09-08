// change this once the API is deployed (see the Dockerfile / README for
// the Hugging Face Spaces deploy steps) - localhost only works while
// running the API on your own machine for testing
const API_BASE = "http://localhost:8000";

const LEVEL_COLORS = {
  "Safe": "#22c55e",
  "Low Risk": "#84cc16",
  "Medium Risk": "#eab308",
  "High Risk": "#f97316",
  "Phishing": "#ef4444",
};

async function main() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const url = tab.url || "";
  document.getElementById("url").textContent = url;

  if (!url.startsWith("http")) {
    document.getElementById("status").textContent = "nothing to check on this page";
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/api/v1/check`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    if (!res.ok) throw new Error("request failed (" + res.status + ")");
    const data = await res.json();

    document.getElementById("status").style.display = "none";
    document.getElementById("body").style.display = "block";

    const levelEl = document.getElementById("level");
    levelEl.textContent = data.level;
    levelEl.style.background = LEVEL_COLORS[data.level] || "#555";
    document.getElementById("score").textContent = (data.score * 100).toFixed(1) + "%";

    const reasons = document.getElementById("reasons");
    data.explanation.forEach((r) => {
      const li = document.createElement("li");
      li.textContent = r;
      reasons.appendChild(li);
    });
  } catch (err) {
    document.getElementById("status").textContent = "could not reach the classifier: " + err.message;
  }
}

main();
