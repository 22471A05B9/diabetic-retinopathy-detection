const fileInput = document.getElementById("file");
const btn = document.getElementById("btnPredict");
const preview = document.getElementById("preview");
const previewEmpty = document.getElementById("previewEmpty");
const lime = document.getElementById("lime");
const limeEmpty = document.getElementById("limeEmpty");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const pDr = document.getElementById("p_dr");
const pNoDr = document.getElementById("p_nodr");
const MAX_BYTES = 10 * 1024 * 1024;

function resetResults() {
  resultEl.textContent = "";
  pDr.textContent = "–";
  pNoDr.textContent = "–";
  lime.hidden = true;
  lime.removeAttribute("src");
  limeEmpty.hidden = false;
}

fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  resetResults();
  statusEl.textContent = "";
  if (!file) { preview.hidden = true; previewEmpty.hidden = false; return; }
  if (!file.type.startsWith("image/")) {
    statusEl.textContent = "Please choose an image file (JPG or PNG).";
    fileInput.value = "";
    return;
  }
  if (file.size > MAX_BYTES) {
    statusEl.textContent = "Image is over 10 MB. Choose a smaller file.";
    fileInput.value = "";
    return;
  }
  if (preview.src) URL.revokeObjectURL(preview.src);
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  previewEmpty.hidden = true;
});

btn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) { statusEl.textContent = "Choose an image first."; return; }
  btn.disabled = true;
  resetResults();
  statusEl.textContent = "Predicting. The LIME explanation can take up to a minute on a CPU…";
  try {
    const fd = new FormData();
    fd.append("image", file);
    const res = await fetch("/predict", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Prediction failed. Try again.");

    const isDr = data.prediction === "DR";
    const badge = document.createElement("span");
    badge.className = "badge " + (isDr ? "dr" : "nodr");
    badge.textContent = data.prediction;
    resultEl.replaceChildren(badge);

    pDr.textContent = Number(data.probabilities?.DR ?? 0).toFixed(4);
    pNoDr.textContent = Number(data.probabilities?.No_DR ?? 0).toFixed(4);

    if (data.lime_url) {
      lime.src = data.lime_url;
      lime.hidden = false;
      limeEmpty.hidden = true;
      statusEl.textContent = "Done. " + (data.disclaimer || "");
    } else {
      statusEl.textContent = "Prediction done, but the LIME explanation failed. " + (data.disclaimer || "");
    }
  } catch (err) {
    console.error(err);
    statusEl.textContent = err.message;
  } finally {
    btn.disabled = false;
  }
});
