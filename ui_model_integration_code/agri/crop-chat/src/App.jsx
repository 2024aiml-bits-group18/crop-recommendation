import React, { useEffect, useRef, useState } from "react";
import "./index.css";
import "./i18n";
import { useTranslation } from "react-i18next";
import i18n, { API_BASE } from "./i18n";

const PREDICT_URL = `${API_BASE}/predict`;

const PHASES = {
  INTRO: "intro",
  MANUAL_DISTRICT: "manualDistrict",
  SLOTS: "slots",
  DONE: "done",
};

/** -------------------------------------------
 *  Slider ranges for numeric soil inputs
 * ------------------------------------------- */
const RANGES = {
  pH:        { min: 0.0,  max: 14.0,   step: 0.1,  def: 6.5, unit: "" },
  EC:        { min: 0.0,  max: 16.0,  step: 0.1,  def: 1.0, unit: "dS/m" },
  OC:        { min: 0.0,  max: 10.0,  step: 0.1,  def: 1.0, unit: "%" },

  "Avail-P": { min: 0,    max: 150,   step: 1,    def: 20,  unit: "ppm" },
  "Exch-K":  { min: 0,    max: 800,   step: 5,    def: 150, unit: "ppm" },
  "Avail-Ca":{ min: 0,    max: 5000,  step: 50,   def: 2000,unit: "ppm" },
  "Avail-Mg":{ min: 0,    max: 1000,  step: 10,   def: 200, unit: "ppm" },
  "Avail-S": { min: 0,    max: 100,   step: 1,    def: 15,  unit: "ppm" },

  "Avail-Zn":{ min: 0.0,  max: 10.0,  step: 0.1,  def: 1.0, unit: "ppm" },
  "Avail-B": { min: 0.0,  max: 5.0,   step: 0.05, def: 0.5, unit: "ppm" },
  "Avail-Fe":{ min: 0,    max: 50,    step: 1,    def: 10,  unit: "ppm" },
  "Avail-Cu":{ min: 0.0,  max: 10.0,  step: 0.1,  def: 1.0, unit: "ppm" },
  "Avail-Mn":{ min: 0,    max: 50,    step: 1,    def: 10,  unit: "ppm" },
};

export default function App() {
  const { t } = useTranslation();
  const [flow, setFlow] = useState(null);
  const [langs, setLangs] = useState(["en"]);

  // conversation state
  const [phase, setPhase] = useState(PHASES.INTRO);
  const [messages, setMessages] = useState([]);
  const [answers, setAnswers] = useState({});
  const [step, setStep] = useState(0);
  const [busy, setBusy] = useState(false);

  // input widgets state
  const [districts, setDistricts] = useState([]);
  const [districtSelect, setDistrictSelect] = useState("");

  const [sliderValue, setSliderValue] = useState(null); // current slider val
  const [choiceOpen, setChoiceOpen] = useState([]);     // chip choices for current slot

  const bottomRef = useRef(null);

  // ---------------- load languages, flow, districts ----------------
  useEffect(() => {
    fetch(`${API_BASE}/i18n/languages`)
      .then((r) => r.json())
      .then((d) => setLangs(d.languages || ["en"]))
      .catch(() => {});

    fetch(`${API_BASE}/flow`)
      .then((r) => r.json())
      .then(setFlow)
      .catch(() => {});

    fetch(`${API_BASE}/meta/districts`)
      .then((r) => r.json())
      .then((d) => {
        const arr = Array.isArray(d?.districts) ? d.districts : [];
        setDistricts(arr.sort((a, b) => String(a).localeCompare(String(b))));
      })
      .catch(() => setDistricts([]));
  }, []);

  // restart when flow/lang changes
  useEffect(() => {
    if (!flow) return;
    startIntro();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [flow, i18n.language]);

  // auto scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // --- helpers to push messages ---
  function pushBot(text) {
    setMessages((m) => [...m, { role: "bot", text }]);
  }
  function pushUser(text) {
    setMessages((m) => [...m, { role: "user", text }]);
  }

  // ---------------- intro script ----------------
  function startIntro() {
    setPhase(PHASES.INTRO);
    setMessages([]);
    setAnswers({});
    setStep(0);
    setBusy(false);
    setDistrictSelect("");

    pushBot(t("intro1"));
    pushBot(t("intro2"));
    pushBot(t("enterYourDistrict"));
    setPhase(PHASES.MANUAL_DISTRICT);
  }

  // ---------------- move into slot collection ----------------
  function startSlots() {
    setPhase(PHASES.SLOTS);
    setStep(0);
    pushBot(t("letsBeginParams"));
    if (flow?.slots?.[0]) {
      const s0 = flow.slots[0];
      pushBot(t(s0.promptKey));
      prepControlForSlot(s0);
    }
  }

  // prepare the right-hand control for a given slot
  function prepControlForSlot(slot) {
    if (slot?.kind === "number") {
      const rg = RANGES[slot.name];
      setSliderValue(rg ? rg.def : 0);
      setChoiceOpen([]);
      return;
    }
    if (slot?.kind === "choice") {
      const arr = t(slot.choiceKey, { returnObjects: true });
      setChoiceOpen(Array.isArray(arr) ? arr : []);
      return;
    }
    setChoiceOpen([]);
  }

  // ---------------- submit to API ----------------
  async function submit(payload) {
    setBusy(true);
    pushBot(t("doneGather"));
    try {
      const body = toModelPayload(payload, i18n.language);
      const resp = await fetch(PREDICT_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!resp.ok) {
        pushBot(t("serverError"));
      } else {
        const data = await resp.json();

        // Prefer enriched format if present
        if (Array.isArray(data.top_k_with_prices) && data.top_k_with_prices.length) {
          const today = new Date().toLocaleDateString(undefined, {
            year: "numeric",
            month: "short",
            day: "numeric",
          });

          // Build an HTML table string so it fits in the chat message list
          const header = `
            <table style="width:100%;border-collapse:collapse;margin-top:8px;">
              <thead>
                <tr>
                  <th style="text-align:left;border-bottom:1px solid #334;padding:8px;">#</th>
                  <th style="text-align:left;border-bottom:1px solid #334;padding:8px;">Crop</th>
                  <th style="text-align:left;border-bottom:1px solid #334;padding:8px;">
                    Min – Max Price (as of ${today})
                  </th>
                </tr>
              </thead>
              <tbody>
          `;

          const rows = data.top_k_with_prices
            .map((item, idx) => {
              const crop = item?.commodity ?? "";
              const minP = item?.min_price ?? null;
              const maxP = item?.max_price ?? null;
              const priceText =
                minP == null && maxP == null
                  ? "N/A"
                  : `${formatPrice(minP)} – ${formatPrice(maxP)}`;
              return `
                <tr>
                  <td style="padding:8px;border-bottom:1px solid #223;">${idx + 1}</td>
                  <td style="padding:8px;border-bottom:1px solid #223;">${escapeHtml(crop)}</td>
                  <td style="padding:8px;border-bottom:1px solid #223;">${priceText}</td>
                </tr>
              `;
            })
            .join("");

          const tableHtml = `${header}${rows}</tbody></table>`;
          pushBot(`**${t("resultTitle")}**${tableHtml}`);
        } else if (Array.isArray(data.top_k) && data.top_k.length) {
          // Fallback to legacy list
          const lines = data.top_k
            .slice(0, 10)
            .map((c, i) => `${i + 1}. ${c}`)
            .join("\n");
          pushBot(`**${t("resultTitle")}**\n${lines}`);
        } else {
          const pred = data.prediction || data.crop || "Unknown";
          pushBot(`**${t("resultTitle")}: ${pred}**`);
        }
      }
    } catch {
      pushBot(t("networkError"));
    } finally {
      setBusy(false);
      setPhase(PHASES.DONE);
    }
  }

  function toModelPayload(a, lang) {
    const num = (k) => (k in a && a[k] !== "" ? Number(a[k]) : null);
    return {
      District: a["District"] ?? null,
      Soil_Type_Standard: a["Soil type"] ?? null,
      Season: a["Season"] ?? null,

      pH: num("pH"),
      EC: num("EC"),
      OC: num("OC"),
      "Avail-P": num("Avail-P"),
      "Exch-K": num("Exch-K"),
      "Avail-Ca": num("Avail-Ca"),
      "Avail-Mg": num("Avail-Mg"),
      "Avail-S": num("Avail-S"),
      "Avail-Zn": num("Avail-Zn"),
      "Avail-B": num("Avail-B"),
      "Avail-Fe": num("Avail-Fe"),
      "Avail-Cu": num("Avail-Cu"),
      "Avail-Mn": num("Avail-Mn"),

      Kharif_rain: num("Kharif_rain"),
      Rabi_rain: num("Rabi_rain"),
      Zaid_rain: num("Zaid_rain"),
      lang,
    };
  }

  // small markdown-style bold
  function bold(s) {
    return String(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  }

  // Helpers for table
  function formatPrice(v) {
    if (v == null || Number.isNaN(Number(v))) return "N/A";
    return new Intl.NumberFormat(undefined, {
      style: "currency",
      currency: "INR",
      maximumFractionDigits: 0,
    }).format(Number(v));
  }
  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  if (!flow) return <div className="center">Loading…</div>;

  const slot = flow.slots?.[step];

  // ---------------- render helpers (controls on the right) ----------------
  function renderDistrictControl() {
    if (districts.length > 0) {
      return (
        <div className="panel">
          <label className="label">{t("enterYourDistrict")}</label>
          <select
            className="select"
            value={districtSelect}
            onChange={(e) => setDistrictSelect(e.target.value)}
            disabled={busy}
          >
            <option value="">{t("select")}</option>
            {districts.map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
          <div className="row">
            <button
              className="btn"
              disabled={!districtSelect || busy}
              onClick={() => {
                pushUser(districtSelect);
                const next = { ...answers, District: districtSelect };
                setAnswers(next);
                pushBot(t("okLetsContinue"));
                startSlots();
              }}
            >
              {t("send")}
            </button>
          </div>
          {/* hint removed per request */}
        </div>
      );
    }
    return (
      <FallbackComposer
        placeholder={t("typeDistrict")}
        disabled={busy}
        onSend={(val) => {
          pushUser(val);
          const next = { ...answers, District: val };
          setAnswers(next);
          pushBot(t("okLetsContinue"));
          startSlots();
        }}
      />
    );
  }

  function renderSlotControl() {
    if (!slot || busy) return null;

    if (slot.kind === "number") {
      const range = RANGES[slot.name] || { min: 0, max: 100, step: 1, def: 0, unit: "" };
      const value = sliderValue ?? range.def;

      return (
        <div className="panel">
		  <label className="label">{t(slot.promptKey)}</label>

		  <div className="sliderContainer">
			<input
			  type="range"
			  className="slider slider--wide"
			  min={range.min}
			  max={range.max}
			  step={range.step}
			  value={value}
			  onChange={(e) => setSliderValue(Number(e.target.value))}
			/>

			{/* Value labels */}
			<div className="sliderLabels">
			  <span className="sliderMin">{range.min}{range.unit ? ` ${range.unit}` : ""}</span>
			  <span className="sliderValueCenter">
				{value}{range.unit ? ` ${range.unit}` : ""}
			  </span>
			  <span className="sliderMax">{range.max}{range.unit ? ` ${range.unit}` : ""}</span>
			</div>
		  </div>

		  <div className="row">
			<button
			  className="btn"
			  onClick={() => {
				const val = value;
				pushUser(String(val));
				const next = { ...answers, [slot.name]: val };
				setAnswers(next);

				const more = step < flow.slots.length - 1;
				if (more) {
				  const s2 = flow.slots[step + 1];
				  setStep(step + 1);
				  pushBot(t(s2.promptKey));
				  setSliderValue(null);
				  prepControlForSlot(s2);
				} else {
				  submit(next);
				}
			  }}
			>
			  {t("send")}
			</button>

			<button
			  className="btn btn--outline"
			  onClick={() => setSliderValue(range.def)}
			>
			  {t("reset")}
			</button>
		  </div>
		</div>

      );
    }

    // choice → chips
    if (slot.kind === "choice" && choiceOpen.length) {
      return (
        <div className="choices">
          {choiceOpen.map((opt) => (
            <button key={opt} className="chip" onClick={() => onSendChoice(opt)}>
              {opt}
            </button>
          ))}
        </div>
      );
    }

    // fallback text
    return (
      <FallbackComposer
        placeholder={t("typeHere")}
        onSend={(val) => onSendTextForSlot(val)}
        disabled={busy}
      />
    );
  }

  function onSendChoice(opt) {
    if (phase !== PHASES.SLOTS || !slot) return;
    pushUser(opt);
    const next = { ...answers, [slot.name]: opt };
    setAnswers(next);

    const more = step < flow.slots.length - 1;
    if (more) {
      const s2 = flow.slots[step + 1];
      setStep(step + 1);
      pushBot(t(s2.promptKey));
      prepControlForSlot(s2);
    } else {
      submit(next);
    }
  }

  function onSendTextForSlot(val) {
    if (phase !== PHASES.SLOTS || !slot) return;
    if (!val.trim()) return;
    pushUser(val);
    const next = { ...answers, [slot.name]: val.trim() };
    setAnswers(next);

    const more = step < flow.slots.length - 1;
    if (more) {
      const s2 = flow.slots[step + 1];
      setStep(step + 1);
      pushBot(t(s2.promptKey));
      prepControlForSlot(s2);
    } else {
      submit(next);
    }
  }

  return (
    <div className="app">
      <header className="app__header">
        <div className="title">{t("appTitle")}</div>
        <div className="lang">
          <select
            value={i18n.language}
            onChange={(e) => i18n.changeLanguage(e.target.value)}
          >
            {langs.map((code) => (
              <option key={code} value={code}>{code}</option>
            ))}
          </select>
          <button className="btn btn--outline" onClick={() => startIntro()}>
            {t("restart")}
          </button>
        </div>
      </header>

      <main className="chat">
        <div className="chat__messages">
          {messages.map((m, i) => (
            <div
              key={i}
              className={`msg msg--${m.role}`}
              dangerouslySetInnerHTML={{ __html: bold(m.text) }}
            />
          ))}
          <div ref={bottomRef} />
        </div>

        {/* right-side controls area */}
        <div className="controls">
          {phase === PHASES.MANUAL_DISTRICT && renderDistrictControl()}
          {phase === PHASES.SLOTS && renderSlotControl()}
        </div>
      </main>
    </div>
  );
}

/** simple text composer used only as a fallback */
function FallbackComposer({ placeholder, onSend, disabled }) {
  const [val, setVal] = useState("");
  return (
    <div className="composer">
      <input
        type="text"
        placeholder={placeholder}
        value={val}
        onChange={(e) => setVal(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && onSend(val)}
        disabled={disabled}
      />
      <button className="btn" onClick={() => onSend(val)} disabled={disabled}>
        Send
      </button>
    </div>
  );
}
