(function () {
    "use strict";

    const DAY_MS = 86400000;
    const OFFSET_MS = 8 * 3600000;
    const THEME_KEY = "capi-dashboard-theme";
    const LABELS = {pending: "查詢中…", ok: "已納入", unsupported: "未支援 OK-s，請更新線體程式",
        offline: "離線／逾時，無法查詢", error: "查詢失敗"};
    let rows = [];
    let followCurrent = true;
    let generation = 0;
    let controller = null;
    let refreshTimer = null;
    let config = null;

    function factoryDate(now = new Date()) {
        return new Date(now.getTime() + OFFSET_MS).toISOString().slice(0, 10);
    }

    function currentShift(now = new Date()) {
        const local = new Date(now.getTime() + OFFSET_MS);
        const minutes = local.getUTCHours() * 60 + local.getUTCMinutes();
        if (minutes < 450) local.setUTCDate(local.getUTCDate() - 1);
        return {date: local.toISOString().slice(0, 10), shift: minutes >= 450 && minutes < 1170 ? "day" : "night"};
    }

    function shiftWindow(date, shift) {
        if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || !["day", "night"].includes(shift)) return null;
        const day = new Date(`${date}T00:00:00+08:00`);
        if (!Number.isFinite(day.getTime()) || factoryDate(day) !== date) return null;
        const start = new Date(day.getTime() + (shift === "day" ? 7.5 : 19.5) * 3600000);
        return {start, end: new Date(start.getTime() + DAY_MS / 2)};
    }

    function validateSelection(selection, now = new Date()) {
        const window = shiftWindow(selection.date, selection.shift);
        if (!window) return "請選擇有效的日期與班別。";
        if (selection.date < factoryDate(new Date(now.getTime() - 29 * DAY_MS))) return "僅能查詢最近 29 天的班別。";
        if (window.start > now) return "該班尚未開始，請選擇當班或已結束的班別。";
        return "";
    }

    function normalizeConfig(raw) {
        if (!raw || !Array.isArray(raw.lines)) throw new Error("線體設定格式不正確");
        const positive = (value, fallback) => Number.isFinite(Number(value)) && Number(value) > 0 ? Number(value) : fallback;
        const refreshSeconds = Math.max(30, positive(raw.refreshIntervalSeconds, 30));
        // Deliberately retain disabled lines and every factory/process zone.
        return {lines: raw.lines, refreshSeconds, timeoutMs: Math.min(30, positive(raw.requestTimeoutSeconds, 8)) * 1000};
    }

    function parseCounts(payload) {
        if (!payload || !Object.prototype.hasOwnProperty.call(payload, "ok_s_count")) return null;
        const total = payload.total;
        const converted = payload.ok_s_count;
        const ng = payload.ng_count == null && total === 0 ? 0 : payload.ng_count;
        if (![total, converted, ng].every(value => Number.isSafeInteger(value) && value >= 0) || ng + converted > total) {
            throw new Error("線體回傳的統計數值不完整或不正確");
        }
        return {total, converted, ng};
    }

    function rates(counts) {
        return {before: counts.total ? (counts.ng + counts.converted) / counts.total * 100 : null,
            after: counts.total ? counts.ng / counts.total * 100 : null,
            drop: counts.total ? counts.converted / counts.total * 100 : null};
    }

    function summarize(items) {
        const valid = items.filter(row => row.state === "ok");
        if (!valid.length) return null;
        const counts = valid.reduce((sum, row) => ({total: sum.total + row.counts.total,
            converted: sum.converted + row.counts.converted, ng: sum.ng + row.counts.ng}), {total: 0, converted: 0, ng: 0});
        return {...counts, ...rates(counts), covered: valid.length};
    }

    async function fetchJSON(url, timeoutMs, signal) {
        const request = new AbortController();
        const abort = () => request.abort();
        if (signal?.aborted) request.abort();
        signal?.addEventListener("abort", abort, {once: true});
        const timer = setTimeout(abort, timeoutMs);
        try {
            const response = await fetch(url, {signal: request.signal, cache: "no-store", credentials: "same-origin",
                headers: {Accept: "application/json"}});
            if (!response.ok) throw Object.assign(new Error(`HTTP ${response.status}`), {status: response.status});
            return await response.json();
        } finally {
            clearTimeout(timer);
            signal?.removeEventListener("abort", abort);
        }
    }

    async function requestLine(line, selection, timeoutMs, signal) {
        let apiUrl;
        try {
            apiUrl = new URL(line.apiUrl);
            if (!["http:", "https:"].includes(apiUrl.protocol)) throw new Error();
        } catch (_) {
            return {state: "error", message: "線體 API 網址不正確"};
        }
        const report = new URL("shift_report", apiUrl);
        report.search = new URLSearchParams({date: selection.date, shift: selection.shift}).toString();
        let payload;
        try {
            payload = await fetchJSON(report.href, timeoutMs, signal);
        } catch (error) {
            if (signal?.aborted) throw error;
            if (error.status === 404 || error.status === 405) return {state: "unsupported"};
            if (error.status || error instanceof SyntaxError) return {state: "error"};
            // Old 404 responses may have no CORS headers. Probe the existing read-only status API.
            try {
                await fetchJSON(apiUrl.href, timeoutMs, signal);
                return {state: "unsupported"};
            } catch (probeError) {
                if (signal?.aborted) throw probeError;
                return {state: "offline"};
            }
        }
        try {
            const counts = parseCounts(payload);
            if (!counts) return {state: "unsupported"};
            if (payload.date !== selection.date || payload.shift !== selection.shift) throw new Error("線體回傳的班別不符");
            return {state: "ok", counts};
        } catch (error) {
            return {state: "error", message: error.message};
        }
    }

    const $ = id => document.getElementById(id);
    const number = value => value.toLocaleString("zh-TW");
    const percent = value => value == null ? "—" : `${value.toFixed(2)}%`;
    const points = value => value == null ? "—" : `${value.toFixed(2)} pp`;

    function element(tag, text, className) {
        const node = document.createElement(tag);
        if (text !== undefined) node.textContent = text;
        if (className) node.className = className;
        return node;
    }

    function lineLabel(line) {
        return `${line.factory || "未設定廠區"} / ${line.line || line.id || "未命名線體"}`;
    }

    function render() {
        const summary = summarize(rows);
        $("oks-total").textContent = summary ? number(summary.total) : "—";
        $("oks-converted").textContent = summary ? number(summary.converted) : "—";
        $("oks-before").textContent = percent(summary?.before);
        $("oks-after").textContent = percent(summary?.after);
        $("oks-drop").textContent = points(summary?.drop);
        const beforeNg = summary ? summary.ng + summary.converted : 0;
        $("oks-relative").textContent = beforeNg ? `相對減少 ${(summary.converted / beforeNg * 100).toFixed(2)}% 的排片` : "百分點（pp）・依投入量加權";
        const pending = rows.filter(row => row.state === "pending").length;
        const covered = summary?.covered || 0;
        $("oks-coverage").textContent = rows.length ? `已納入 ${covered} / ${rows.length} 線${pending ? `・${pending} 線查詢中` : covered < rows.length ? "・部分資料，未取得者不計入合計" : "・全部資料已取得"}` : "尚無線體資料";
        $("oks-coverage").dataset.partial = String(covered < rows.length);
        const sort = $("oks-sort").value;
        const sorted = [...rows].sort((a, b) => {
            if (sort !== "line") {
                const value = row => row.state === "ok" ? (sort === "converted" ? row.counts.converted : rates(row.counts).drop ?? -1) : -2;
                const difference = value(b) - value(a);
                if (difference) return difference;
            }
            return lineLabel(a.line).localeCompare(lineLabel(b.line), "zh-TW", {numeric: true}) ||
                String(a.line.pcName || a.line.id).localeCompare(String(b.line.pcName || b.line.id), "zh-TW", {numeric: true});
        });
        const maxDrop = Math.max(0, ...rows.filter(row => row.state === "ok").map(row => rates(row.counts).drop || 0));
        const body = $("oks-lines");
        body.replaceChildren();
        if (!rows.length) {
            const row = element("tr");
            const cell = element("td", "尚無線體資料；請檢查線體設定或重新查詢。", "oks-empty");
            cell.colSpan = 7;
            row.append(cell);
            body.append(row);
        }
        for (const item of sorted) {
            const row = element("tr");
            const identity = element("td");
            identity.append(element("strong", lineLabel(item.line), "oks-line-name"));
            let host = "";
            try { host = new URL(item.line.apiUrl).host; } catch (_) {}
            const info = [item.line.pcName, host, item.line.enabled === false ? "設定已停用・仍納入查詢" : ""].filter(Boolean).join(" · ");
            identity.append(element("span", info, "oks-line-info"));
            row.append(identity);
            const counts = item.state === "ok" ? item.counts : null;
            const rate = counts ? rates(counts) : {};
            row.append(element("td", counts ? number(counts.total) : "—"),
                element("td", counts ? number(counts.converted) : "—", "oks-positive"),
                element("td", percent(rate.before)), element("td", percent(rate.after)));
            const drop = element("td", points(rate.drop), "oks-drop-cell");
            if (rate.drop != null) {
                const bar = element("div", undefined, "oks-bar");
                bar.setAttribute("aria-hidden", "true");
                const fill = element("span");
                fill.style.width = `${maxDrop ? rate.drop / maxDrop * 100 : 0}%`;
                bar.append(fill);
                drop.append(bar);
                drop.title = `${counts.converted} ÷ ${counts.total} × 100 = ${points(rate.drop)}`;
            }
            row.append(drop);
            const status = element("td");
            const badge = element("span", item.message || (counts?.total === 0 ? "已納入・無投入" : LABELS[item.state]), "oks-badge");
            badge.dataset.state = item.state;
            status.append(badge);
            row.append(status);
            body.append(row);
        }
    }

    function setCurrent() {
        const selection = currentShift();
        $("oks-date").value = selection.date;
        $("oks-shift").value = selection.shift;
    }

    function localTimestamp(date) {
        return new Date(date.getTime() + OFFSET_MS).toISOString().slice(0, 16).replace("T", " ");
    }

    async function refresh() {
        const token = ++generation;
        clearTimeout(refreshTimer);
        controller?.abort();
        controller = new AbortController();
        const signal = controller.signal;
        const now = new Date();
        if (followCurrent) setCurrent();
        const selection = {date: $("oks-date").value, shift: $("oks-shift").value};
        $("oks-current").setAttribute("aria-pressed", String(followCurrent));
        $("oks-date").max = factoryDate(now);
        $("oks-date").min = factoryDate(new Date(now.getTime() - 29 * DAY_MS));
        const invalid = validateSelection(selection, now);
        rows = [];
        render();
        $("oks-error").hidden = !invalid;
        $("oks-error").textContent = invalid;
        $("oks-range").textContent = "";
        if (invalid) {
            $("oks-updated").textContent = "尚未查詢";
            return;
        }
        const window = shiftWindow(selection.date, selection.shift);
        $("oks-range").textContent = `${localTimestamp(window.start)} ～ ${localTimestamp(window.end)} (UTC+8)`;
        $("oks-updated").textContent = "讀取全部線體清單…";
        try {
            const directFile = location.protocol === "file:";
            const loadedConfig = normalizeConfig(directFile ? globalThis.CAPI_DASHBOARD_CONFIG :
                await fetchJSON("/api/central-dashboard/ok-s-config", 8000, signal));
            if (token !== generation) return;
            config = loadedConfig;
            $("oks-scope").textContent = directFile ? "config.js 全部線體・含停用線體（直接開檔模式）" : "全部廠區・全部製程・包含停用線體";
            rows = config.lines.map(line => ({line, state: "pending"}));
            render();
            $("oks-updated").textContent = "查詢各線班報中…";
            await Promise.all(rows.map(async row => {
                const result = await requestLine(row.line, selection, config.timeoutMs, signal);
                if (token !== generation) return;
                Object.assign(row, result);
                render();
            }));
            if (token !== generation) return;
            const updated = new Date(Date.now() + OFFSET_MS).toISOString().slice(11, 19);
            $("oks-updated").textContent = `更新於 ${updated}${followCurrent ? `・每 ${config.refreshSeconds} 秒更新` : "・指定班別"}`;
        } catch (error) {
            if (token !== generation) return;
            $("oks-error").textContent = `無法取得完整線體清單：${error.message || error}。請確認中控主機已更新，再重新查詢。`;
            $("oks-error").hidden = false;
            $("oks-updated").textContent = "查詢失敗";
        } finally {
            if (token === generation && followCurrent) refreshTimer = setTimeout(refresh, (config?.refreshSeconds || 30) * 1000);
        }
    }

    function syncTheme() {
        const dark = document.documentElement.dataset.theme === "dark";
        $("oks-theme").textContent = dark ? "淺色模式" : "深色模式";
        $("oks-theme").setAttribute("aria-pressed", String(dark));
    }

    function initialize() {
        syncTheme();
        $("oks-theme").addEventListener("click", () => {
            const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
            document.documentElement.dataset.theme = theme;
            try { localStorage.setItem(THEME_KEY, theme); } catch (_) {}
            syncTheme();
        });
        $("oks-query-form").addEventListener("submit", event => { event.preventDefault(); refresh(); });
        for (const id of ["oks-date", "oks-shift"]) {
            $(id).addEventListener("change", () => { followCurrent = false; refresh(); });
        }
        $("oks-current").addEventListener("click", () => { followCurrent = true; refresh(); });
        $("oks-sort").addEventListener("change", render);
        refresh();
    }

    if (typeof module !== "undefined" && module.exports) {
        module.exports = {currentShift, shiftWindow, validateSelection, normalizeConfig, parseCounts, rates, summarize, requestLine};
    }
    if (typeof document !== "undefined") document.addEventListener("DOMContentLoaded", initialize);
})();
