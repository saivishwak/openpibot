import "./styles.css";

import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import * as Tooltip from "@radix-ui/react-tooltip";
import { App } from "./App";

let savedTheme: string | null = null;
try {
  savedTheme = window.localStorage.getItem("openpibot.theme");
} catch {
  savedTheme = null;
}
const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
document.documentElement.classList.toggle(
  "dark",
  savedTheme === "dark" || ((savedTheme === null || savedTheme === "system") && prefersDark),
);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Tooltip.Provider delayDuration={250}>
        <App />
      </Tooltip.Provider>
    </BrowserRouter>
  </React.StrictMode>,
);
