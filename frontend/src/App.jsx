import { useState } from "react";
import "./App.css";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

const YES_NO = [
  ["SMOKING", "Smoking"],
  ["FINGER_DISCOLORATION", "Finger discoloration"],
  ["MENTAL_STRESS", "Mental stress"],
  ["EXPOSURE_TO_POLLUTION", "Exposure to pollution"],
  ["LONG_TERM_ILLNESS", "Long-term illness"],
  ["IMMUNE_WEAKNESS", "Immune weakness"],
  ["BREATHING_ISSUE", "Breathing issue"],
  ["ALCOHOL_CONSUMPTION", "Alcohol consumption"],
  ["THROAT_DISCOMFORT", "Throat discomfort"],
  ["CHEST_TIGHTNESS", "Chest tightness"],
  ["FAMILY_HISTORY", "Family history"],
  ["SMOKING_FAMILY_HISTORY", "Smoking in family"],
  ["STRESS_IMMUNE", "Stress-related immune issue"],
];

const initial = {
  AGE: 50, GENDER: 0, ENERGY_LEVEL: 55, OXYGEN_SATURATION: 96,
  ...Object.fromEntries(YES_NO.map(([k]) => [k, 0])),
};

export default function App() {
  const [form, setForm] = useState(initial);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));

  async function submit(e) {
    e.preventDefault();
    setLoading(true); setError(""); setResult(null);
    try {
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      setResult(await res.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="wrap">
      <h1>Lung Disease Risk Predictor</h1>
      <p className="muted">Educational ML demo (scikit-learn + FastAPI). Not medical advice.</p>

      <form onSubmit={submit}>
        <div className="grid">
          <label>Age
            <input type="number" min="1" max="120" value={form.AGE}
              onChange={(e) => set("AGE", +e.target.value)} />
          </label>
         <label>Gender
            <select value={form.GENDER} onChange={(e) => set("GENDER", +e.target.value)}>
              <option value={0}>Female</option>
              <option value={1}>Male</option>
            </select>
          </label>
          <label>Energy level: {form.ENERGY_LEVEL}
            <input type="range" min="0" max="100" value={form.ENERGY_LEVEL}
              onChange={(e) => set("ENERGY_LEVEL", +e.target.value)} />
          </label>
          <label>Oxygen saturation (%)
            <input type="number" step="0.1" min="50" max="100" value={form.OXYGEN_SATURATION}
              onChange={(e) => set("OXYGEN_SATURATION", +e.target.value)} />
          </label>
        </div>

        <div className="checks">
          {YES_NO.map(([k, label]) => (
            <label key={k} className="check">
              <input type="checkbox" checked={!!form[k]}
                onChange={(e) => set(k, e.target.checked ? 1 : 0)} />
              {label}
            </label>
          ))}
        </div>

        <button disabled={loading}>{loading ? "Predicting..." : "Predict"}</button>
      </form>

      {error && <p className="error">Error: {error} (the free API may be waking up, try again in ~30s)</p>}

      {result && (
        <section className={`result ${result.risk_level}`}>
          <h2>{(result.probability * 100).toFixed(1)}% probability</h2>
          <p>Prediction: <b>{result.prediction}</b> | Risk level: <b>{result.risk_level}</b></p>
          <small>{result.disclaimer}</small>
        </section>
      )}
    </main>
  );
}
