import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import HttpBackend from "i18next-http-backend";

// Backend API base (use LAN IP for phone testing, see notes below)
export const API_BASE = "http://localhost:8000/api";

i18n
  .use(HttpBackend)
  .use(initReactI18next)
  .init({
    lng: "en",
    fallbackLng: "en",
    interpolation: { escapeValue: false },
    backend: {
      loadPath: `${API_BASE}/i18n/{{lng}}`,
    },
  });

export default i18n;
