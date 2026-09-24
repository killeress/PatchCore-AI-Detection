(function () {
    "use strict";

    const DEFAULT_REFRESH_SECONDS = 30;
    const MIN_REFRESH_SECONDS = 30;
    const DEFAULT_TIMEOUT_SECONDS = 8;
    const THEME_STORAGE_KEY = "capi-dashboard-theme";
    // 歷史班報可查詢上限：線體 OK 紀錄保留 30 天，預留 1 天清理邊界
    const HISTORY_MAX_LOOKBACK_DAYS = 29;
    const HEALTH_THRESHOLDS = Object.freeze({
        diskFreeWarningPercent: 15,
        diskFreeCriticalPercent: 10,
        ramUsedWarningPercent: 85,
        ramUsedCriticalPercent: 95,
        vramUsedWarningPercent: 85,
        vramUsedCriticalPercent: 95,
        gpuTemperatureWarningC: 80,
        gpuTemperatureCriticalC: 90
    });
    let config = normalizeConfig(window.CAPI_DASHBOARD_CONFIG);
    const lineStates = new Map();
    let activeProcessZone = "capi";

    let refreshTimer = null;
    let countdownTimer = null;
    let clockTimer = null;
    let nextRefreshAt = null;
    let isRefreshing = false;
    let activeMode = "realtime";
    let historyInitialized = false;
    let historyShift = "day";

    document.addEventListener("DOMContentLoaded", initialize);

    async function initialize() {
        initializeTheme();
        // 關掉瀏覽器的捲動位置還原，避免刷新後被拉回頂端（干擾下方跳過 banner 的動作）
        if ("scrollRestoration" in window.history) {
            window.history.scrollRestoration = "manual";
        }
        const directFileMode = window.location.protocol === "file:";
        const helpLink = document.getElementById("dashboard-help-link");
        if (helpLink) helpLink.hidden = directFileMode;
        if (!directFileMode) {
            try {
                const response = await fetch("/api/central-dashboard/config", {
                    method: "GET",
                    headers: { "Accept": "application/json" },
                    cache: "no-store",
                    credentials: "same-origin"
                });
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                config = normalizeConfig(await response.json());
            } catch (error) {
                showConfigError(
                    `SQLite 設定讀取失敗，暫時使用 config.js 備援：${error.message || error}`
                );
            }
        }

        document.title = config.title;
        setText(document.getElementById("dashboard-title"), config.title);
        const settingsLink = document.getElementById("dashboard-settings-link");
        if (settingsLink) {
            settingsLink.hidden = directFileMode;
        }
        initializeModeTabs();

        const activeLines = config.lines.filter((line) => line.enabled !== false);
        if (activeLines.length === 0) {
            showConfigError("尚未設定任何啟用中的線體。");
            updateSummary();
            startClock();
            skipDashboardBanner();
            return;
        }

        const seenIds = new Set();
        const factoryGrids = new Map();
        for (const line of activeLines) {
            if (!line.id || seenIds.has(line.id)) {
                showConfigError("每條線都必須有不重複的 ID，請至設備設定頁檢查。");
                continue;
            }
            seenIds.add(line.id);
            const factory = line.factory || "未設定廠別";
            if (!factoryGrids.has(factory)) {
                factoryGrids.set(factory, createFactorySection(factory, factoryGrids.size + 1));
            }
            createLineCard(line, factoryGrids.get(factory));
        }
        initializeProcessTabs();
        updateSummary();
        skipDashboardBanner();

        startClock();
        countdownTimer = window.setInterval(updateRefreshStatus, 1000);
        refreshAllLines();
    }

    function skipDashboardBanner() {
        // 內容渲染完成後再跳過頂部 banner：讓標題列對齊視窗頂（帶 # 錨點時保留瀏覽器跳轉）
        if (!window.location.hash) {
            const topbar = document.querySelector(".topbar");
            if (topbar) {
                // 空設定時內容較短，仍需保留一個視窗高度，才能完整捲過 banner。
                const main = document.querySelector("main");
                if (main) {
                    main.style.minHeight = `calc(100vh - ${topbar.offsetHeight}px)`;
                }
                window.scrollTo({
                    top: topbar.getBoundingClientRect().top + window.scrollY,
                    behavior: "instant"
                });
            }
        }
    }

    function normalizeConfig(rawConfig) {
        const raw = rawConfig && typeof rawConfig === "object" ? rawConfig : {};
        const refreshIntervalSeconds = Math.max(
            MIN_REFRESH_SECONDS,
            toPositiveInteger(raw.refreshIntervalSeconds, DEFAULT_REFRESH_SECONDS)
        );
        const requestTimeoutSeconds = Math.min(
            refreshIntervalSeconds - 1,
            Math.max(3, toPositiveInteger(raw.requestTimeoutSeconds, DEFAULT_TIMEOUT_SECONDS))
        );

        return {
            title: String(raw.title || "寧波廠區 CAPI AI 中控看板"),
            refreshIntervalSeconds,
            requestTimeoutSeconds,
            lines: Array.isArray(raw.lines) ? raw.lines : [],
            watchModels: normalizeWatchModels(raw.watchModels)
        };
    }

    function normalizeWatchModels(value) {
        if (!Array.isArray(value)) {
            return [];
        }
        const seen = new Set();
        const models = [];
        for (const item of value) {
            const code = String(item || "").trim().toUpperCase();
            if (code && !seen.has(code)) {
                seen.add(code);
                models.push(code);
            }
        }
        return models;
    }

    function toPositiveInteger(value, fallback) {
        const number = Number(value);
        return Number.isFinite(number) && number > 0 ? Math.round(number) : fallback;
    }

    function createFactorySection(factory, index) {
        const section = document.createElement("section");
        const headingId = `factory-title-${index}`;
        section.className = "factory-group";
        section.setAttribute("aria-labelledby", headingId);

        const heading = document.createElement("div");
        heading.className = "factory-heading";

        const eyebrow = document.createElement("span");
        eyebrow.className = "factory-heading-label";
        eyebrow.textContent = "FACTORY ZONE";

        const title = document.createElement("h3");
        title.id = headingId;
        title.textContent = factory;

        const lineGrid = document.createElement("div");
        lineGrid.className = "line-grid";
        lineGrid.setAttribute("aria-label", `${factory} 線體狀態`);

        heading.append(eyebrow, title);
        section.append(heading, lineGrid);
        document.getElementById("factory-sections").appendChild(section);
        return lineGrid;
    }

    function createLineCard(line, lineGrid) {
        const template = document.getElementById("line-card-template");
        const card = template.content.firstElementChild.cloneNode(true);
        const overviewRow = createOverviewRow(line);
        const processZone = String(line.line || "").trim().toUpperCase().startsWith("AAPI")
            ? "aapi"
            : "capi";

        card.dataset.lineId = line.id;
        card.dataset.processZone = processZone;
        setField(card, "pc-name", line.pcName || line.id);
        setField(card, "line-name", line.line || "未設定線體");

        configureLink(card, "dashboard", line.dashboardUrl || deriveBaseUrl(line.apiUrl));

        lineGrid.appendChild(card);
        lineStates.set(line.id, {
            line,
            card,
            overviewRow,
            processZone,
            status: "checking",
            data: null,
            error: ""
        });
        renderLineCard(lineStates.get(line.id));
    }

    function initializeProcessTabs() {
        const tabs = document.getElementById("process-tabs");
        const availableZones = new Set(
            Array.from(lineStates.values(), (state) => state.processZone)
        );
        activeProcessZone = availableZones.has("capi")
            ? "capi"
            : (availableZones.values().next().value || "capi");
        tabs.hidden = availableZones.size < 2;
        for (const tab of tabs.querySelectorAll(".process-tab")) {
            tab.addEventListener("click", () => selectProcessZone(tab.dataset.processZone));
        }
        selectProcessZone(activeProcessZone);
    }

    function selectProcessZone(zoneId) {
        activeProcessZone = zoneId;
        for (const tab of document.querySelectorAll(".process-tab")) {
            const selected = tab.dataset.processZone === activeProcessZone;
            tab.setAttribute("aria-pressed", String(selected));
        }

        for (const state of lineStates.values()) {
            const hidden = state.processZone !== activeProcessZone;
            state.overviewRow.hidden = hidden;
            state.card.hidden = hidden;
        }
        for (const group of document.querySelectorAll(".factory-group")) {
            group.hidden = !Array.from(group.querySelectorAll(".line-card")).some(
                (card) => !card.hidden
            );
        }
        updateSummary();
        renderAlerts();
        // 歷史班報模式下切換製程類別：結果表跟著重查
        if (activeMode === "history") {
            refreshHistoryReport();
        }
    }

    // ── 歷史班報模式 ────────────────────────────────────────

    function initializeModeTabs() {
        document.body.dataset.mode = activeMode;
        const tabs = document.getElementById("mode-tabs");
        if (!tabs) {
            return;
        }
        for (const tab of tabs.querySelectorAll("[data-mode]")) {
            tab.addEventListener("click", () => selectMode(tab.dataset.mode));
        }
    }

    function selectMode(mode) {
        if (mode !== "history") {
            mode = "realtime";
        }
        if (mode === activeMode) {
            return;
        }
        activeMode = mode;
        document.body.dataset.mode = mode;
        for (const tab of document.querySelectorAll("#mode-tabs [data-mode]")) {
            tab.setAttribute("aria-pressed", String(tab.dataset.mode === mode));
        }
        const historySection = document.getElementById("history-section");
        if (historySection) {
            historySection.hidden = mode !== "history";
        }
        if (mode === "history") {
            clearTimeout(refreshTimer);
            refreshTimer = null;
            nextRefreshAt = null;
            initializeHistoryMode();
            refreshHistoryReport();
        } else {
            refreshAllLines();
        }
    }

    function initializeHistoryMode() {
        if (historyInitialized) {
            return;
        }
        historyInitialized = true;
        const dateInput = document.getElementById("history-date");
        if (!dateInput) {
            return;
        }
        const now = new Date();
        dateInput.max = formatDateValue(now);
        const minDate = new Date(now);
        minDate.setDate(minDate.getDate() - HISTORY_MAX_LOOKBACK_DAYS);
        dateInput.min = formatDateValue(minDate);

        const fallback = latestCompletedShift(now);
        dateInput.value = fallback.date;
        selectHistoryShift(fallback.shift);

        dateInput.addEventListener("change", refreshHistoryReport);
        for (const btn of document.querySelectorAll("#history-shift-tabs [data-shift]")) {
            btn.addEventListener("click", () => {
                selectHistoryShift(btn.dataset.shift);
                refreshHistoryReport();
            });
        }
    }

    function selectHistoryShift(shift) {
        historyShift = shift === "night" ? "night" : "day";
        for (const btn of document.querySelectorAll("#history-shift-tabs [data-shift]")) {
            btn.setAttribute("aria-pressed", String(btn.dataset.shift === historyShift));
        }
    }

    function formatDateValue(date) {
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, "0");
        const day = String(date.getDate()).padStart(2, "0");
        return `${year}-${month}-${day}`;
    }

    function latestCompletedShift(now) {
        const minutes = now.getHours() * 60 + now.getMinutes();
        const today = formatDateValue(now);
        const yesterdayDate = new Date(now);
        yesterdayDate.setDate(yesterdayDate.getDate() - 1);
        const yesterday = formatDateValue(yesterdayDate);
        if (minutes >= 19 * 60 + 30) {
            return { date: today, shift: "day" };
        }
        if (minutes >= 7 * 60 + 30) {
            return { date: yesterday, shift: "night" };
        }
        return { date: yesterday, shift: "day" };
    }

    function isShiftSelectable(dateValue, shift, now = new Date()) {
        if (!dateValue) {
            return false;
        }
        const today = formatDateValue(now);
        const minDate = new Date(now);
        minDate.setDate(minDate.getDate() - HISTORY_MAX_LOOKBACK_DAYS);
        if (dateValue < formatDateValue(minDate) || dateValue > today) {
            return false;
        }
        if (dateValue === today) {
            if (shift === "night") {
                return false;
            }
            return now.getHours() * 60 + now.getMinutes() >= 19 * 60 + 30;
        }
        return true;
    }

    function updateHistoryShiftAvailability() {
        const dateInput = document.getElementById("history-date");
        if (!dateInput) {
            return;
        }
        for (const btn of document.querySelectorAll("#history-shift-tabs [data-shift]")) {
            const selectable = isShiftSelectable(dateInput.value, btn.dataset.shift);
            btn.disabled = !selectable;
            btn.title = selectable ? "" : "該班尚未結束，請改用即時模式查看";
        }
    }

    function computeHistoryRange(dateValue, shift) {
        const parts = String(dateValue || "").split("-").map(Number);
        if (parts.length !== 3 || parts.some((n) => !Number.isFinite(n))) {
            return null;
        }
        const start = new Date(parts[0], parts[1] - 1, parts[2], 7, 30);
        const end = new Date(parts[0], parts[1] - 1, parts[2], 19, 30);
        if (shift === "night") {
            start.setHours(19, 30);
            end.setDate(end.getDate() + 1);
            end.setHours(7, 30);
        }
        const label = (d) =>
            `${d.getFullYear()}/${String(d.getMonth() + 1).padStart(2, "0")}/` +
            `${String(d.getDate()).padStart(2, "0")} ` +
            `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
        return { start, end, text: `${label(start)} ~ ${label(end)}` };
    }

    async function refreshHistoryReport() {
        const dateInput = document.getElementById("history-date");
        const rangeElement = document.getElementById("history-range");
        const tbody = document.getElementById("history-overview");
        if (!dateInput || !tbody || !rangeElement) {
            return;
        }
        updateHistoryShiftAvailability();
        const dateValue = dateInput.value;
        if (!isShiftSelectable(dateValue, historyShift)) {
            tbody.innerHTML = "";
            setText(rangeElement, "該班尚未結束，請改用即時模式查看");
            return;
        }
        const range = computeHistoryRange(dateValue, historyShift);
        setText(rangeElement, range ? `統計區間：${range.text}` : "");

        tbody.innerHTML = "";
        // 歷史班報列出全部線體（含停用；廠區網段過濾已在設定 API 端處理），
        // 並跟著「製程類別」頁籤過濾 CAPI/AAPI
        const lines = historyLinesForActiveZone();
        if (lines.length === 0) {
            const row = document.createElement("tr");
            const cell = document.createElement("td");
            cell.colSpan = 4;
            cell.className = "history-message";
            cell.textContent = "此製程類別尚未設定任何線體。";
            row.appendChild(cell);
            tbody.appendChild(row);
            return;
        }

        const rowRefs = lines.map((line) => {
            const row = document.createElement("tr");
            row.dataset.lineId = line.id || "";
            const lineCell = document.createElement("td");
            lineCell.className = "overview-line-cell";
            const identity = document.createElement("div");
            identity.className = "overview-line";
            const nameGroup = document.createElement("span");
            nameGroup.className = "overview-line-name";
            const lineName = document.createElement("strong");
            lineName.textContent = line.line || "未設定線體";
            nameGroup.appendChild(lineName);
            identity.appendChild(nameGroup);
            lineCell.appendChild(identity);
            row.appendChild(lineCell);
            tbody.appendChild(row);
            showHistoryRowMessage(row, "查詢中…", "muted");
            return { line, row };
        });

        await Promise.all(
            rowRefs.map(({ line, row }) => fetchHistoryForLine(line, row, dateValue, historyShift))
        );
    }

    function historyLinesForActiveZone() {
        return config.lines.filter((line) => {
            const zone = String(line.line || "").trim().toUpperCase().startsWith("AAPI")
                ? "aapi"
                : "capi";
            return zone === activeProcessZone;
        });
    }

    function appendHistoryPlaceholderCell(row, text, className) {
        const cell = document.createElement("td");
        const span = document.createElement("span");
        span.className = className;
        setText(span, text);
        cell.appendChild(span);
        row.appendChild(cell);
    }

    function showHistoryRowMessage(row, message, tone) {
        // 訊息小圓章放線體名稱右邊，資料欄比照即時總覽顯示「AOI — / AI — / —」
        const identity = row.querySelector(".overview-line");
        for (const oldChip of row.querySelectorAll(".history-state-chip")) {
            oldChip.remove();
        }
        while (row.cells.length > 1) {
            row.deleteCell(1);
        }
        // 比照即時總覽：離線列淡紅底
        row.dataset.state = tone === "error" ? "offline" : "";
        const chip = document.createElement("span");
        chip.className = "history-state-chip";
        chip.dataset.tone = tone;
        chip.textContent = message;
        identity.appendChild(chip);
        appendHistoryPlaceholderCell(row, "AOI —", "overview-rate overview-rate-aoi");
        appendHistoryPlaceholderCell(row, "AI —", "overview-rate overview-rate-ai");
        appendHistoryPlaceholderCell(row, "—", "overview-total");
    }

    function renderHistoryRowData(row, payload) {
        const total = numberValue(payload.total);
        const aoiNg = optionalNumber(payload.aoi_ng_count);
        const aiNg = optionalNumber(payload.ng_count ?? payload.ai_ng_count);

        while (row.cells.length > 1) {
            row.deleteCell(1);
        }
        row.dataset.state = "";
        for (const oldChip of row.querySelectorAll(".history-state-chip")) {
            oldChip.remove();
        }

        // 比照即時總覽：AOI 琥珀、AI 青、投入等寬粗體
        const appendRateCell = (label, ngCount, rateClass) => {
            const cell = document.createElement("td");
            cell.className = "overview-rate-cell";
            const span = document.createElement("span");
            span.className = `overview-rate ${rateClass}`;
            const hasRate = ngCount !== null && total > 0;
            setText(span, `${label} ${hasRate ? formatRate(ngCount, total) : "—"}`);
            span.title = hasRate
                ? `${label} 該班排片率：${formatRate(ngCount, total)}（NG ${formatNumber(ngCount)} / 總投入 ${formatNumber(total)}）`
                : `${label} 該班尚無可計算資料`;
            cell.appendChild(span);
            row.appendChild(cell);
        };

        appendRateCell("AOI", aoiNg, "overview-rate-aoi");
        appendRateCell("AI", aiNg, "overview-rate-ai");

        const totalCell = document.createElement("td");
        totalCell.className = "overview-total-cell";
        const totalSpan = document.createElement("span");
        totalSpan.className = "overview-total";
        setText(totalSpan, formatNumber(total));
        totalSpan.title = "該班投入 = OK + NG + ERR";
        totalCell.appendChild(totalSpan);
        row.appendChild(totalCell);
    }

    async function fetchHistoryForLine(line, row, dateValue, shift) {
        const baseUrl = deriveBaseUrl(line.apiUrl);
        if (!baseUrl) {
            showHistoryRowMessage(row, "離線無法查詢", "error");
            return;
        }
        const controller = new AbortController();
        const timeoutId = window.setTimeout(
            () => controller.abort(),
            config.requestTimeoutSeconds * 1000
        );
        try {
            const response = await fetch(
                `${baseUrl}api/shift_report?date=${encodeURIComponent(dateValue)}&shift=${encodeURIComponent(shift)}`,
                {
                    method: "GET",
                    headers: { "Accept": "application/json" },
                    cache: "no-store",
                    credentials: "omit",
                    signal: controller.signal
                }
            );
            if (response.status === 404) {
                showHistoryRowMessage(row, "未更新，請更新線體程式", "warning");
                return;
            }
            if (!response.ok) {
                showHistoryRowMessage(row, "離線無法查詢", "error");
                return;
            }
            const payload = await response.json();
            if (!payload || typeof payload !== "object") {
                showHistoryRowMessage(row, "離線無法查詢", "error");
                return;
            }
            renderHistoryRowData(row, payload);
        } catch (_error) {
            // 舊版線體對不存在的路徑回 404 時不帶 CORS 標頭，瀏覽器讀不到狀態碼，
            // 與真正離線無法區分；改探既有的 /api/status（有 CORS）：
            // 探得到 = 線體在線但程序未更新；探不到 = 真的離線
            const reachable = await probeLineReachable(line);
            if (reachable) {
                showHistoryRowMessage(row, "未更新，請更新線體程式", "warning");
            } else {
                showHistoryRowMessage(row, "離線無法查詢", "error");
            }
        } finally {
            clearTimeout(timeoutId);
        }
    }

    async function probeLineReachable(line) {
        const controller = new AbortController();
        const timeoutId = window.setTimeout(
            () => controller.abort(),
            config.requestTimeoutSeconds * 1000
        );
        try {
            const response = await fetch(line.apiUrl, {
                method: "GET",
                headers: { "Accept": "application/json" },
                cache: "no-store",
                credentials: "omit",
                signal: controller.signal
            });
            return response.ok;
        } catch (_error) {
            return false;
        } finally {
            clearTimeout(timeoutId);
        }
    }

    function createOverviewRow(line) {
        const row = document.createElement("tr");
        row.dataset.lineId = line.id;
        row.dataset.state = "checking";
        if (line.isProduction === true) {
            row.dataset.production = "true";
        }

        const lineCell = document.createElement("td");
        lineCell.className = "overview-line-cell";
        const lineIdentity = document.createElement("div");
        lineIdentity.className = "overview-line";
        const lineNameGroup = document.createElement("span");
        lineNameGroup.className = "overview-line-name";
        const watchBadge = document.createElement("span");
        watchBadge.className = "overview-watch-badge";
        watchBadge.dataset.field = "overview-watch";
        watchBadge.textContent = "★";
        watchBadge.hidden = true;
        lineNameGroup.appendChild(watchBadge);
        const lineName = document.createElement("strong");
        lineName.textContent = line.line || "未設定線體";
        lineNameGroup.appendChild(lineName);
        lineIdentity.appendChild(lineNameGroup);
        if (line.isProduction === true) {
            const badge = document.createElement("span");
            badge.className = "overview-production-badge";
            badge.textContent = "上線";
            badge.title = "正式上線設備";
            lineIdentity.appendChild(badge);
        }
        const switchBadges = document.createElement("span");
        switchBadges.className = "overview-switch-badges";
        switchBadges.dataset.field = "overview-switch-badges";
        lineIdentity.appendChild(switchBadges);
        lineCell.appendChild(lineIdentity);

        const ipCell = document.createElement("td");
        ipCell.className = "overview-ip-cell";
        const ip = document.createElement("code");
        ip.className = "overview-ip";
        ip.textContent = extractHostname(line.apiUrl) || "—";
        ip.title = line.apiUrl || "";
        ipCell.appendChild(ip);

        const statusCell = document.createElement("td");
        statusCell.className = "overview-status-cell";
        const status = document.createElement("span");
        status.className = "status-pill overview-status";
        status.dataset.field = "overview-status";
        status.textContent = statusText("checking");
        statusCell.appendChild(status);

        const aoiCell = document.createElement("td");
        aoiCell.className = "overview-aoi-cell";
        const aoi = document.createElement("span");
        aoi.className = "overview-aoi";
        aoi.dataset.field = "overview-aoi";
        aoi.dataset.state = "unknown";
        aoi.textContent = "AOI —";
        aoiCell.appendChild(aoi);

        const aoiRateCell = document.createElement("td");
        aoiRateCell.className = "overview-rate-cell overview-aoi-rate-cell";
        const aoiRate = document.createElement("span");
        aoiRate.className = "overview-rate overview-rate-aoi";
        aoiRate.dataset.field = "aoi-rate";
        aoiRate.textContent = "AOI —";
        aoiRateCell.appendChild(aoiRate);

        const aiRateCell = document.createElement("td");
        aiRateCell.className = "overview-rate-cell overview-ai-rate-cell";
        const aiRate = document.createElement("span");
        aiRate.className = "overview-rate overview-rate-ai";
        aiRate.dataset.field = "ai-rate";
        aiRate.textContent = "AI —";
        aiRateCell.appendChild(aiRate);

        const totalCell = document.createElement("td");
        totalCell.className = "overview-total-cell";
        const total = document.createElement("span");
        total.className = "overview-total";
        total.dataset.field = "overview-total";
        total.textContent = "—";
        total.title = "當班投入 = OK + NG + ERR";
        totalCell.appendChild(total);

        const activityCell = document.createElement("td");
        activityCell.className = "overview-activity-cell";
        const activity = document.createElement("span");
        activity.className = "overview-activity";
        activity.dataset.field = "overview-activity";
        activity.textContent = "尚無判定";
        activityCell.appendChild(activity);

        const alertCell = document.createElement("td");
        alertCell.className = "overview-alert-cell is-empty";
        const alerts = document.createElement("div");
        alerts.className = "overview-alerts";
        alerts.dataset.field = "overview-alerts";
        alertCell.appendChild(alerts);

        const updateCell = document.createElement("td");
        updateCell.className = "overview-update-cell";
        const updateInfo = document.createElement("div");
        updateInfo.className = "overview-update";
        updateInfo.dataset.field = "overview-update";
        updateInfo.dataset.state = "unknown";
        const updateBadge = document.createElement("span");
        updateBadge.className = "overview-update-badge";
        updateBadge.dataset.field = "overview-update-badge";
        updateBadge.hidden = true;
        const version = document.createElement("span");
        version.className = "overview-version";
        version.dataset.field = "overview-version";
        version.textContent = "—";
        const updateButton = document.createElement("button");
        updateButton.type = "button";
        updateButton.className = "overview-update-action";
        updateButton.dataset.action = "update";
        updateButton.textContent = "更新程式";
        updateButton.hidden = true;
        updateButton.addEventListener("click", () => applyCentralUpdate(line.id));
        updateInfo.append(updateBadge, version, updateButton);
        updateCell.appendChild(updateInfo);

        const linkCell = document.createElement("td");
        linkCell.className = "overview-link-cell";
        const link = document.createElement("a");
        link.className = "overview-link";
        link.dataset.link = "dashboard";
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = "開啟";
        linkCell.appendChild(link);

        row.append(
            lineCell,
            ipCell,
            statusCell,
            aoiCell,
            aoiRateCell,
            aiRateCell,
            totalCell,
            activityCell,
            alertCell,
            updateCell,
            linkCell
        );
        const dashboardUrl = line.dashboardUrl || deriveBaseUrl(line.apiUrl);
        configureLink(row, "dashboard", dashboardUrl);
        document.getElementById("line-overview").appendChild(row);
        return row;
    }

    function configureLink(root, name, url) {
        const element = root && root.querySelector
            ? root.querySelector(`[data-link="${name}"]`)
            : null;
        if (!element) {
            return;
        }
        if (!url) {
            element.hidden = true;
            element.removeAttribute("href");
            return;
        }
        element.href = url;
    }

    async function refreshAllLines() {
        if (isRefreshing || lineStates.size === 0 || activeMode !== "realtime") {
            return;
        }

        isRefreshing = true;
        nextRefreshAt = null;
        updateRefreshStatus();

        await Promise.all(Array.from(lineStates.values(), refreshLine));

        isRefreshing = false;
        updateSummary();
        renderAlerts();
        scheduleNextRefresh();
    }

    async function refreshLine(state) {
        const controller = new AbortController();
        const timeoutId = window.setTimeout(
            () => controller.abort(),
            config.requestTimeoutSeconds * 1000
        );

        try {
            const response = await fetch(state.line.apiUrl, {
                method: "GET",
                headers: { "Accept": "application/json" },
                cache: "no-store",
                credentials: "omit",
                signal: controller.signal
            });

            if (!response.ok) {
                throw new Error(`API 回應 HTTP ${response.status}`);
            }

            const rawData = await response.json();
            if (!rawData || typeof rawData !== "object") {
                throw new Error("API 回傳內容不是 JSON 物件");
            }

            state.data = normalizeStatus(rawData);
            if (!state.data.running) {
                state.status = "warning";
                state.error = "API 可連線，但服務回報未運行。";
            } else if (state.data.gpuHealth.active) {
                state.status = "warning";
                state.error = state.data.gpuHealth.title;
            } else if (
                state.data.lineActivity.available &&
                state.data.lineActivity.isHalted
            ) {
                // 停線：連線正常但最近窗口產量 ≤ 閾值（線體端算好）
                state.status = "halted";
                state.error = "";
            } else {
                state.status = "online";
                state.error = "";
            }
        } catch (error) {
            state.status = "offline";
            state.error = readableFetchError(error);
        } finally {
            clearTimeout(timeoutId);
            renderLineCard(state);
            // 每條線回覆後立刻重算平均排片率，不等整輪（避免被離線線體逾時拖慢）
            renderSummaryAverages();
        }
    }

    function normalizeStatus(raw) {
        const server = asObject(raw.server);
        const traffic = asObject(raw.traffic);
        const stats = asObject(raw.stats);
        const latestEvent = asObject(raw.latest_event);
        const hardware = asObject(raw.hardware || server.hardware);
        const gpu = asObject(hardware.gpu);
        const gpuHealth = asObject(raw.gpu_health);
        const memory = asObject(hardware.memory || hardware.ram);
        const disk = asObject(hardware.disk);
        const update = asObject(raw.update);
        const lineActivity = asObject(raw.line_activity);
        const modelSwitchAlert = asObject(raw.model_switch_alert);

        return {
            running: server.running !== false,
            hostname: textValue(server.hostname),
            uptime: textValue(server.uptime),
            modelVersion: textValue(server.model_version),
            device: textValue(server.device || gpu.name),
            total: numberValue(stats.total_requests ?? stats.total),
            ok: numberValue(stats.total_ok ?? stats.ok_count),
            ng: numberValue(stats.total_ng ?? stats.ng_count),
            aoiNg: optionalNumber(stats.aoi_ng_count),
            aiNg: optionalNumber(stats.ai_ng_count ?? stats.total_ng ?? stats.ng_count),
            err: numberValue(stats.total_err ?? stats.err_count),
            overexposed: optionalNumber(stats.overexposed_count),
            avgTime: optionalNumber(
                stats.avg_time ??
                stats.average_processing_seconds ??
                asObject(raw.performance).avg_seconds
            ),
            activeConnections: numberValue(traffic.active_connections),
            connectedMachines: Array.isArray(traffic.connected_machines)
                ? traffic.connected_machines
                : [],
            latestEvent: {
                glassId: textValue(latestEvent.glass_id),
                modelId: textValue(latestEvent.model_id),
                machineNo: textValue(latestEvent.machine_no),
                judgment: textValue(latestEvent.judgment || latestEvent.detail),
                time: textValue(latestEvent.time),
                duration: textValue(latestEvent.duration)
            },
            update: {
                status: textValue(update.status).toLowerCase() || "unknown",
                currentVersion: textValue(update.current_version),
                pendingVersion: textValue(update.pending_version),
                canApply: update.can_apply === true,
                failureReason: textValue(update.failure_reason),
                centralApplySupported: update.central_apply_supported === true
            },
            hardware: {
                vramUsedGb: optionalNumber(gpu.vram_used_gb),
                vramTotalGb: optionalNumber(gpu.vram_total_gb),
                gpuUtilization: optionalNumber(gpu.utilization_percent),
                gpuTemperature: optionalNumber(gpu.temperature_c),
                ramUsedGb: optionalNumber(memory.used_gb),
                ramTotalGb: optionalNumber(memory.total_gb),
                ramUsedPercent: optionalNumber(memory.used_percent),
                diskFreeGb: optionalNumber(disk.free_gb),
                diskTotalGb: optionalNumber(disk.total_gb)
            },
            gpuHealth: {
                active: gpuHealth.active === true,
                severity: gpuHealth.severity === "critical" ? "critical" : "warning",
                title: textValue(gpuHealth.title),
                message: textValue(gpuHealth.message)
            },
            lineActivity: {
                available: raw.line_activity !== undefined && raw.line_activity !== null,
                windowMinutes: optionalNumber(lineActivity.window_minutes),
                panelCount: optionalNumber(lineActivity.panel_count),
                haltThreshold: optionalNumber(lineActivity.halt_threshold),
                isHalted: lineActivity.is_halted === true
            },
            modelSwitches: (Array.isArray(modelSwitchAlert.events) ? modelSwitchAlert.events : [])
                .map((event) => {
                    const item = asObject(event);
                    return {
                        machineNo: textValue(item.machine_no),
                        previousModel: textValue(item.previous_model),
                        newModel: textValue(item.new_model),
                        switchedAt: textValue(item.switched_at),
                        expiresAt: textValue(item.expires_at)
                    };
                })
                .filter((event) => event.newModel)
        };
    }

    function asObject(value) {
        return value && typeof value === "object" && !Array.isArray(value) ? value : {};
    }

    function textValue(value) {
        return value === undefined || value === null ? "" : String(value).trim();
    }

    function numberValue(value) {
        const number = Number(value);
        return Number.isFinite(number) ? number : 0;
    }

    function optionalNumber(value) {
        if (value === undefined || value === null || value === "") {
            return null;
        }
        const number = Number(value);
        return Number.isFinite(number) ? number : null;
    }

    function matchedWatchModel(state) {
        if (!state || (state.status !== "online" && state.status !== "halted") || !state.data) {
            return "";
        }
        const modelId = state.data.latestEvent.modelId;
        if (!modelId) {
            return "";
        }
        return config.watchModels.includes(modelId.toUpperCase())
            ? modelId
            : "";
    }

    function updateWatchBadge(badge, state) {
        if (!badge) {
            return;
        }
        const matched = matchedWatchModel(state);
        badge.hidden = !matched;
        if (matched) {
            badge.title = `正在生產關注機種：${matched}`;
        } else {
            badge.removeAttribute("title");
        }
    }

    function updateSwitchBadges(container, state) {
        if (!container) {
            return;
        }
        container.replaceChildren();
        if (!state.data || state.status === "offline") {
            return;
        }
        const events = state.data.modelSwitches;
        if (!events.length) {
            return;
        }
        // 徽章只放四個字；切換時間、新舊機種、機台等詳情收進 tooltip
        const badge = document.createElement("span");
        badge.className = "line-switch-badge";
        badge.textContent = events.length > 1 ? `機種切換×${events.length}` : "機種切換";
        badge.title = events
            .map((event) => {
                const hhmm = event.switchedAt.length >= 16
                    ? event.switchedAt.slice(11, 16)
                    : event.switchedAt;
                return `${hhmm} 機台 ${event.machineNo || "未知"}：${event.previousModel || "?"}→${event.newModel}（提醒至 ${event.expiresAt}）`;
            })
            .join("\n");
        container.appendChild(badge);
    }

    function renderLineCard(state) {
        const card = state.card;
        const data = state.data;
        renderOverviewRow(state);
        card.dataset.state = state.status;
        const healthAlerts = data && state.status !== "offline"
            ? getHardwareAlerts(data)
            : [];
        card.dataset.health = healthAlerts.length ? healthAlerts[0].severity : "normal";
        setField(card, "status", statusText(state.status));
        updateWatchBadge(card.querySelector('[data-field="watch-badge"]'), state);
        updateSwitchBadges(card.querySelector('[data-field="switch-badges"]'), state);

        const errorElement = card.querySelector('[data-field="error"]');
        errorElement.hidden = !state.error;
        setText(errorElement, state.error);

        if (!data) {
            return;
        }

        setField(
            card,
            "pc-name",
            data.hostname ||
                state.line.pcName ||
                extractHostname(state.line.apiUrl) ||
                state.line.id
        );

        const validCount = data.ok + data.ng;
        const totalForErrorRate = data.total || validCount + data.err;

        setField(card, "total", formatNumber(data.total));
        setField(card, "ok-count", formatNumber(data.ok));
        setField(card, "ng-count", formatNumber(data.ng));
        setField(card, "err-count", formatNumber(data.err));
        setField(card, "ok-rate", formatRate(data.ok, validCount));
        setField(card, "ng-rate", formatRate(data.ng, validCount));
        setField(card, "err-rate", formatRate(data.err, totalForErrorRate));
        setField(
            card,
            "overexposed",
            data.overexposed === null ? "—" : formatNumber(data.overexposed)
        );
        setField(
            card,
            "avg-time",
            data.avgTime === null ? "—" : `${formatDecimal(data.avgTime, 1)}s`
        );
        setField(
            card,
            "latest-duration",
            `最近：${data.latestEvent.duration || "—"}`
        );
        setField(card, "model-version", data.modelVersion || "—");
        setField(card, "device", data.device || "—");
        setField(card, "vram", formatVram(data.hardware));
        setField(card, "gpu-health", data.gpuHealth.active ? data.gpuHealth.title : formatGpuHealth(data.hardware));
        setField(card, "ram", formatMemory(data.hardware));
        setField(card, "disk", formatDisk(data.hardware));
        setField(
            card,
            "connections",
            `${formatNumber(data.activeConnections)} / ${formatNumber(data.connectedMachines.length)} 機台`
        );

        const latestLabel = [data.latestEvent.glassId, data.latestEvent.machineNo]
            .filter(Boolean)
            .join(" / ");
        setField(card, "latest-glass", latestLabel || "尚無資料");
        setField(card, "latest-judgment", displayJudgment(data.latestEvent.judgment));
        setField(card, "latest-time", data.latestEvent.time || "—");
        updateJudgmentClass(card, data.latestEvent.judgment);
    }

    function renderOverviewRow(state) {
        const row = state.overviewRow;
        if (!row) {
            return;
        }
        row.dataset.state = state.status;
        updateWatchBadge(row.querySelector('[data-field="overview-watch"]'), state);
        updateSwitchBadges(row.querySelector('[data-field="overview-switch-badges"]'), state);
        const status = row.querySelector('[data-field="overview-status"]');
        setText(status, statusText(state.status));
        status.title = state.error || statusText(state.status);

        const data = state.data;
        const aoi = row.querySelector('[data-field="overview-aoi"]');
        if (!data || state.status === "checking") {
            setText(aoi, "AOI —");
            aoi.dataset.state = "unknown";
            aoi.title = "正在讀取 AOI 連線狀態";
        } else if (state.status === "offline") {
            setText(aoi, "AOI —");
            aoi.dataset.state = "unknown";
            aoi.title = "設備離線，無法確認 AOI 連線狀態";
        } else if (data.activeConnections > 0) {
            setText(aoi, `AOI ${formatNumber(data.activeConnections)}`);
            aoi.dataset.state = "connected";
            aoi.title = `${formatNumber(data.activeConnections)} 個 AOI 即時連線`;
        } else {
            setText(aoi, "AOI 未連線");
            aoi.dataset.state = "disconnected";
            aoi.title = "目前沒有 AOI 即時連線";
        }

        if (data && state.status !== "offline") {
            renderOverviewRejectRate(row, "aoi-rate", "AOI", data.aoiNg, data.total);
            renderOverviewRejectRate(row, "ai-rate", "AI", data.aiNg, data.total);
        } else {
            renderOverviewRejectRate(row, "aoi-rate", "AOI", null, 0);
            renderOverviewRejectRate(row, "ai-rate", "AI", null, 0);
        }

        const totalEl = row.querySelector('[data-field="overview-total"]');
        if (data && state.status !== "offline" && state.status !== "checking") {
            setText(totalEl, formatNumber(data.total));
        } else {
            setText(totalEl, "—");
        }

        renderOverviewActivity(state);
        renderOverviewUpdate(state);

        const alertCell = row.querySelector(".overview-alert-cell");
        const alertContainer = row.querySelector('[data-field="overview-alerts"]');
        const healthAlerts = data && state.status !== "offline"
            ? getHardwareAlerts(data)
            : [];
        row.dataset.health = healthAlerts.length
            ? healthAlerts[0].severity
            : "normal";
        alertContainer.replaceChildren();
        for (const alert of healthAlerts) {
            const badge = document.createElement("span");
            badge.className = `overview-alert overview-alert-${alert.severity}`;
            badge.textContent = `⚠ ${alert.summary}`;
            badge.title = alert.message;
            alertContainer.appendChild(badge);
        }
        alertCell.classList.toggle("is-empty", healthAlerts.length === 0);
    }

    function renderOverviewUpdate(state) {
        const row = state.overviewRow;
        const container = row.querySelector('[data-field="overview-update"]');
        const badge = row.querySelector('[data-field="overview-update-badge"]');
        const version = row.querySelector('[data-field="overview-version"]');
        const updateButton = row.querySelector('[data-action="update"]');
        const update = state.data && state.data.update;
        const currentVersion = update && update.currentVersion !== "unknown"
            ? update.currentVersion
            : "";
        const pendingVersion = update ? update.pendingVersion : "";
        const updateStatus = update ? update.status : "unknown";
        const canApplyUpdate = Boolean(
            update &&
            update.canApply &&
            update.centralApplySupported &&
            state.status !== "offline" &&
            state.status !== "checking"
        );

        badge.hidden = true;
        updateButton.hidden = true;
        updateButton.disabled = false;
        updateButton.textContent = "更新程式";
        container.dataset.state = "unknown";
        setText(version, currentVersion || "—");

        if (updateStatus === "apply_requested" || updateStatus === "installing") {
            container.dataset.state = "applying";
            badge.hidden = false;
            setText(badge, "更新中");
            setText(version, pendingVersion || currentVersion || "—");
            container.title = "設備正在套用更新並重新啟動，完成後版本會自動刷新。";
            return;
        }

        if (updateStatus === "failed") {
            container.dataset.state = "failed";
            badge.hidden = false;
            setText(badge, "更新失敗");
            setText(version, pendingVersion || currentVersion || "—");
            updateButton.hidden = !canApplyUpdate;
            updateButton.setAttribute(
                "aria-label",
                `${state.line.line || state.line.id} 重試更新至版本 ${pendingVersion || "未知"}`
            );
            container.title = update.failureReason || "請開啟設備更新頁查看失敗狀態。";
            return;
        }

        if (pendingVersion) {
            container.dataset.state = "pending";
            badge.hidden = false;
            setText(badge, "新版本");
            setText(version, pendingVersion);
            updateButton.hidden = !canApplyUpdate;
            updateButton.setAttribute(
                "aria-label",
                `${state.line.line || state.line.id} 更新至版本 ${pendingVersion}`
            );
            if (!update.centralApplySupported) {
                container.title = "設備需先安裝支援中央更新的版本一次。";
            } else if (state.status === "offline") {
                container.title = `目前版本 ${currentVersion || "未知"}，設備離線，恢復連線後可更新至 ${pendingVersion}。`;
            } else {
                container.title = `目前版本 ${currentVersion || "未知"}，點擊後在此頁確認並直接啟動設備更新。`;
            }
            return;
        }

        if (currentVersion) {
            container.dataset.state = "current";
            container.title = `目前程式版本 ${currentVersion}`;
            return;
        }

        container.title = "此設備尚未提供程式版本與更新狀態。";
    }

    async function applyCentralUpdate(lineId) {
        const state = lineStates.get(lineId);
        const update = state && state.data && state.data.update;
        if (
            !state ||
            !update ||
            !update.pendingVersion ||
            !update.canApply ||
            !update.centralApplySupported
        ) {
            return;
        }

        const lineName = state.line.line || state.line.id;
        const confirmed = window.confirm(
            `即將直接更新 ${lineName} 至版本 ${update.pendingVersion}，並重新啟動該設備服務。\n\n` +
            "請確認該設備已停止檢測，且目前沒有玻璃正在推論或訓練工作進行中。"
        );
        if (!confirmed) {
            return;
        }

        const updateButton = state.overviewRow.querySelector('[data-action="update"]');
        updateButton.disabled = true;
        updateButton.textContent = "啟動中…";
        try {
            const response = await fetch("/api/central-dashboard/update/apply", {
                method: "POST",
                headers: {
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                },
                credentials: "same-origin",
                body: JSON.stringify({
                    lineId: state.line.id,
                    expectedVersion: update.pendingVersion
                })
            });
            const payload = await response.json();
            if (response.status === 401) {
                window.location.href = "/settings/login?next=%2Fcentral_dashboard%2F";
                return;
            }
            if (!response.ok) {
                throw new Error(payload.error || `HTTP ${response.status}`);
            }

            update.status = "apply_requested";
            update.canApply = false;
            renderLineCard(state);
            window.setTimeout(refreshAllLines, 3000);
        } catch (error) {
            window.alert(`無法啟動 ${lineName} 更新：${error.message || error}`);
            renderOverviewUpdate(state);
        }
    }

    function renderOverviewRejectRate(row, field, label, ngCount, total) {
        const element = row.querySelector(`[data-field="${field}"]`);
        const hasRate = ngCount !== null && total > 0;
        const rate = hasRate ? formatRate(ngCount, total) : "—";
        setText(element, `${label} ${rate}`);
        element.title = hasRate
            ? `${label} 當班排片率：${rate}（NG ${formatNumber(ngCount)} / 總投入 ${formatNumber(total)}）`
            : `${label} 當班尚無可計算資料`;
    }

    function renderOverviewActivity(state, now = new Date()) {
        const row = state.overviewRow;
        if (!row) {
            return;
        }
        const activity = row.querySelector('[data-field="overview-activity"]');
        const latestEvent = state.data && state.data.latestEvent;
        if (!latestEvent || !latestEvent.judgment) {
            setText(activity, "尚無判定");
            activity.title = "";
            return;
        }

        const judgment = displayJudgment(latestEvent.judgment);
        const relativeTime = formatRelativeTime(latestEvent.time, now);
        setText(
            activity,
            relativeTime ? `最近 ${judgment} · ${relativeTime}` : `最近 ${judgment}`
        );
        activity.title = latestEvent.time ? `API 判定時間：${latestEvent.time}` : "";
    }

    function statusText(status) {
        return {
            checking: "連線中",
            online: "正常",
            halted: "停線",
            warning: "服務異常",
            offline: "離線"
        }[status] || "未知";
    }

    function formatNumber(value) {
        return Math.round(Number(value) || 0).toLocaleString("zh-TW");
    }

    function formatRate(value, denominator) {
        if (!denominator) {
            return "—";
        }
        return `${formatDecimal((value / denominator) * 100, 1)}%`;
    }

    function formatDecimal(value, digits) {
        return Number(value).toLocaleString("zh-TW", {
            minimumFractionDigits: digits,
            maximumFractionDigits: digits
        });
    }

    function formatRelativeTime(value, now = new Date()) {
        const eventTime = parseEventTime(value, now);
        if (!eventTime) {
            return textValue(value);
        }

        const elapsedSeconds = Math.max(
            0,
            Math.floor((now.getTime() - eventTime.getTime()) / 1000)
        );
        if (elapsedSeconds < 5) {
            return "剛剛";
        }
        if (elapsedSeconds < 60) {
            return `${elapsedSeconds} 秒前`;
        }

        const elapsedMinutes = Math.floor(elapsedSeconds / 60);
        if (elapsedMinutes < 60) {
            return `${elapsedMinutes} 分鐘前`;
        }

        const elapsedHours = Math.floor(elapsedMinutes / 60);
        if (elapsedHours < 24) {
            return `${elapsedHours} 小時前`;
        }
        return `${Math.floor(elapsedHours / 24)} 天前`;
    }

    function parseEventTime(value, now) {
        const rawValue = textValue(value);
        if (!rawValue) {
            return null;
        }

        const timeOnly = /^(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?$/.exec(rawValue);
        if (timeOnly) {
            const hours = Number(timeOnly[1]);
            const minutes = Number(timeOnly[2]);
            const seconds = Number(timeOnly[3]);
            const milliseconds = Number((timeOnly[4] || "").padEnd(3, "0")) || 0;
            if (hours > 23 || minutes > 59 || seconds > 59) {
                return null;
            }

            const eventTime = new Date(now);
            eventTime.setHours(hours, minutes, seconds, milliseconds);
            if (eventTime.getTime() - now.getTime() > 5 * 60 * 1000) {
                eventTime.setDate(eventTime.getDate() - 1);
            }
            return eventTime;
        }

        const normalizedValue = rawValue.replace(
            /^(\d{4}-\d{2}-\d{2})\s+/,
            "$1T"
        );
        const eventTime = new Date(normalizedValue);
        return Number.isNaN(eventTime.getTime()) ? null : eventTime;
    }

    function formatVram(hardware) {
        if (hardware.vramUsedGb === null || hardware.vramTotalGb === null) {
            return "—";
        }
        return `${formatDecimal(hardware.vramUsedGb, 1)} / ${formatDecimal(hardware.vramTotalGb, 1)} GB`;
    }

    function formatGpuHealth(hardware) {
        const parts = [];
        if (hardware.gpuUtilization !== null) {
            parts.push(`${formatDecimal(hardware.gpuUtilization, 0)}%`);
        }
        if (hardware.gpuTemperature !== null) {
            parts.push(`${formatDecimal(hardware.gpuTemperature, 0)}°C`);
        }
        return parts.length ? parts.join(" / ") : "—";
    }

    function formatMemory(hardware) {
        if (hardware.ramUsedGb !== null && hardware.ramTotalGb !== null) {
            const usage = `${formatDecimal(hardware.ramUsedGb, 1)} / ${formatDecimal(hardware.ramTotalGb, 1)} GB`;
            return hardware.ramUsedPercent === null
                ? usage
                : `${usage} (${formatDecimal(hardware.ramUsedPercent, 0)}%)`;
        }
        return hardware.ramUsedPercent === null
            ? "—"
            : `${formatDecimal(hardware.ramUsedPercent, 0)}%`;
    }

    function formatDisk(hardware) {
        if (hardware.diskFreeGb === null) {
            return "—";
        }
        if (hardware.diskTotalGb === null) {
            return `${formatDecimal(hardware.diskFreeGb, 1)} GB`;
        }
        return `${formatDecimal(hardware.diskFreeGb, 1)} / ${formatDecimal(hardware.diskTotalGb, 1)} GB`;
    }

    function displayJudgment(value) {
        const judgment = String(value || "").toUpperCase();
        if (judgment.startsWith("ERR:HY")) {
            return "HY";
        }
        if (judgment.startsWith("NG")) {
            return "NG";
        }
        if (judgment === "OK-I") {
            return "OK-i";
        }
        if (judgment.startsWith("OK")) {
            return "OK";
        }
        if (judgment.startsWith("ERR")) {
            return "ERR";
        }
        return judgment || "—";
    }

    function updateJudgmentClass(card, value) {
        const badge = card.querySelector('[data-field="latest-judgment"]');
        badge.classList.remove("is-ok", "is-ng", "is-err");
        const judgment = displayJudgment(value);
        if (judgment === "OK" || judgment === "OK-i") {
            badge.classList.add("is-ok");
        } else if (judgment === "NG") {
            badge.classList.add("is-ng");
        } else if (judgment === "ERR" || judgment === "HY") {
            badge.classList.add("is-err");
        }
    }

    function getHardwareAlerts(data) {
        const alerts = [];
        const hardware = data.hardware || {};
        if (data.gpuHealth && data.gpuHealth.active) {
            alerts.push({
                severity: data.gpuHealth.severity,
                summary: data.gpuHealth.title,
                message: `${data.gpuHealth.title}：${data.gpuHealth.message}`
            });
        }

        if (hardware.diskFreeGb !== null && hardware.diskTotalGb > 0) {
            const freePercent = (hardware.diskFreeGb / hardware.diskTotalGb) * 100;
            if (freePercent <= HEALTH_THRESHOLDS.diskFreeCriticalPercent) {
                alerts.push({
                    severity: "critical",
                    summary: `硬碟 ${formatDecimal(freePercent, 0)}%`,
                    message: `硬碟空間嚴重不足：剩餘 ${formatDecimal(freePercent, 1)}%（${formatDecimal(hardware.diskFreeGb, 1)} / ${formatDecimal(hardware.diskTotalGb, 1)} GB）`
                });
            } else if (freePercent <= HEALTH_THRESHOLDS.diskFreeWarningPercent) {
                alerts.push({
                    severity: "warning",
                    summary: `硬碟 ${formatDecimal(freePercent, 0)}%`,
                    message: `硬碟空間偏低：剩餘 ${formatDecimal(freePercent, 1)}%（${formatDecimal(hardware.diskFreeGb, 1)} / ${formatDecimal(hardware.diskTotalGb, 1)} GB）`
                });
            }
        }

        if (hardware.ramUsedPercent !== null) {
            if (hardware.ramUsedPercent >= HEALTH_THRESHOLDS.ramUsedCriticalPercent) {
                alerts.push({
                    severity: "critical",
                    summary: `RAM ${formatDecimal(hardware.ramUsedPercent, 0)}%`,
                    message: `RAM 使用率過高：${formatDecimal(hardware.ramUsedPercent, 0)}%`
                });
            } else if (hardware.ramUsedPercent >= HEALTH_THRESHOLDS.ramUsedWarningPercent) {
                alerts.push({
                    severity: "warning",
                    summary: `RAM ${formatDecimal(hardware.ramUsedPercent, 0)}%`,
                    message: `RAM 使用率偏高：${formatDecimal(hardware.ramUsedPercent, 0)}%`
                });
            }
        }

        if (hardware.vramUsedGb !== null && hardware.vramTotalGb > 0) {
            const usedPercent = (hardware.vramUsedGb / hardware.vramTotalGb) * 100;
            if (usedPercent >= HEALTH_THRESHOLDS.vramUsedCriticalPercent) {
                alerts.push({
                    severity: "critical",
                    summary: `VRAM ${formatDecimal(usedPercent, 0)}%`,
                    message: `VRAM 使用率過高：${formatDecimal(usedPercent, 0)}%（${formatDecimal(hardware.vramUsedGb, 1)} / ${formatDecimal(hardware.vramTotalGb, 1)} GB）`
                });
            } else if (usedPercent >= HEALTH_THRESHOLDS.vramUsedWarningPercent) {
                alerts.push({
                    severity: "warning",
                    summary: `VRAM ${formatDecimal(usedPercent, 0)}%`,
                    message: `VRAM 使用率偏高：${formatDecimal(usedPercent, 0)}%（${formatDecimal(hardware.vramUsedGb, 1)} / ${formatDecimal(hardware.vramTotalGb, 1)} GB）`
                });
            }
        }

        if (hardware.gpuTemperature !== null) {
            if (hardware.gpuTemperature >= HEALTH_THRESHOLDS.gpuTemperatureCriticalC) {
                alerts.push({
                    severity: "critical",
                    summary: `GPU ${formatDecimal(hardware.gpuTemperature, 0)}°C`,
                    message: `GPU 溫度過高：${formatDecimal(hardware.gpuTemperature, 0)}°C`
                });
            } else if (hardware.gpuTemperature >= HEALTH_THRESHOLDS.gpuTemperatureWarningC) {
                alerts.push({
                    severity: "warning",
                    summary: `GPU ${formatDecimal(hardware.gpuTemperature, 0)}°C`,
                    message: `GPU 溫度偏高：${formatDecimal(hardware.gpuTemperature, 0)}°C`
                });
            }
        }

        return alerts.sort((left, right) => {
            const priority = { critical: 0, warning: 1 };
            return priority[left.severity] - priority[right.severity];
        });
    }

    function updateSummary() {
        const states = Array.from(lineStates.values()).filter(
            (state) => state.processZone === activeProcessZone
        );

        setText(document.getElementById("summary-total-lines"), states.length);
        setText(
            document.getElementById("summary-online"),
            states.filter((state) => state.status === "online").length
        );
        setText(
            document.getElementById("summary-warning"),
            states.filter((state) => state.status === "warning").length
        );
        setText(
            document.getElementById("summary-offline"),
            states.filter((state) => state.status === "offline").length
        );
        renderSummaryAverages();
    }

    function renderSummaryAverages() {
        // 跟著「製程類別」頁籤：目前類別中「上線」且狀態「正常」線體的
        // 當班排片率算術平均；排片率無資料（顯示 —）的線自動排除
        const aoiElement = document.getElementById("summary-avg-aoi-rate");
        const aiElement = document.getElementById("summary-avg-ai-rate");
        if (!aoiElement || !aiElement) {
            return;
        }
        const zoneLabel = activeProcessZone.toUpperCase();
        const eligible = Array.from(lineStates.values()).filter(
            (state) =>
                state.processZone === activeProcessZone &&
                state.line.isProduction === true &&
                state.status === "online" &&
                state.data
        );
        const renderAverage = (element, label, pickNg) => {
            const entries = [];
            for (const state of eligible) {
                const ng = pickNg(state.data);
                if (ng === null || !(state.data.total > 0)) {
                    continue;
                }
                entries.push({
                    name: state.line.line || state.line.id,
                    rate: (ng / state.data.total) * 100
                });
            }
            if (entries.length === 0) {
                setText(element, "—");
                element.title = `${label}：無符合條件的 ${zoneLabel} 線體（需上線、狀態正常且有當班資料）`;
                return;
            }
            const mean =
                entries.reduce((sum, entry) => sum + entry.rate, 0) / entries.length;
            setText(element, `${formatDecimal(mean, 1)}%`);
            element.title =
                `${label}（${entries.length} 條線算術平均）：` +
                entries
                    .map((entry) => `${entry.name} ${formatDecimal(entry.rate, 1)}%`)
                    .join("、");
        };
        renderAverage(aoiElement, `${zoneLabel} 平均 AOI 排片率`, (data) => data.aoiNg);
        renderAverage(aiElement, `${zoneLabel} 平均 AI 排片率`, (data) => data.aiNg);
    }

    function renderAlerts() {
        const alerts = [];
        for (const state of lineStates.values()) {
            if (state.processZone !== activeProcessZone) {
                continue;
            }
            const label = `${state.line.factory || "未設定廠別"} / ${state.line.line || state.line.id}`;
            if (state.status === "offline") {
                alerts.push({
                    severity: "critical",
                    message: `${label}：${state.error || statusText(state.status)}`
                });
            } else if (state.status === "warning" && !(state.data && state.data.running && state.data.gpuHealth.active)) {
                alerts.push({
                    severity: "warning",
                    message: `${label}：${state.error || statusText(state.status)}`
                });
            }

            if (state.data && state.status !== "offline") {
                for (const alert of getHardwareAlerts(state.data)) {
                    alerts.push({
                        severity: alert.severity,
                        message: `${label}：${alert.message}`
                    });
                }
            }
        }
        const panel = document.getElementById("alert-panel");
        const list = document.getElementById("alert-list");
        list.replaceChildren();
        const hasCritical = alerts.some((alert) => alert.severity === "critical");
        panel.classList.toggle("has-critical", hasCritical);
        panel.classList.toggle("has-warning", !hasCritical && alerts.length > 0);

        if (alerts.length === 0) {
            panel.hidden = true;
            return;
        }

        for (const alert of alerts) {
            const item = document.createElement("li");
            item.className = `alert-item alert-${alert.severity}`;
            item.textContent = alert.message;
            list.appendChild(item);
        }
        panel.hidden = false;
    }

    function scheduleNextRefresh() {
        clearTimeout(refreshTimer);
        if (activeMode !== "realtime") {
            nextRefreshAt = null;
            updateRefreshStatus();
            return;
        }
        nextRefreshAt = Date.now() + config.refreshIntervalSeconds * 1000;
        refreshTimer = window.setTimeout(refreshAllLines, config.refreshIntervalSeconds * 1000);
        updateRefreshStatus();
    }

    function startClock() {
        updateClock();
        clearInterval(clockTimer);
        clockTimer = window.setInterval(function () {
            updateClock();
        }, 1000);
    }

    function updateClock() {
        const now = new Date();
        setText(
            document.getElementById("current-date"),
            new Intl.DateTimeFormat("zh-TW", {
                year: "numeric",
                month: "2-digit",
                day: "2-digit",
                weekday: "short"
            }).format(now)
        );
        setText(
            document.getElementById("current-time"),
            new Intl.DateTimeFormat("zh-TW", {
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
                hour12: false
            }).format(now)
        );
        for (const state of lineStates.values()) {
            renderOverviewActivity(state, now);
        }
    }

    function updateRefreshStatus() {
        const element = document.getElementById("refresh-status");
        if (isRefreshing) {
            element.dataset.state = "refreshing";
            setText(element, "正在更新各 PC");
            return;
        }
        if (!nextRefreshAt) {
            element.dataset.state = "ready";
            setText(element, "準備更新");
            return;
        }
        element.dataset.state = "countdown";
        const seconds = Math.max(0, Math.ceil((nextRefreshAt - Date.now()) / 1000));
        setText(element, `${seconds} 秒後更新`);
    }

    function readableFetchError(error) {
        if (error && error.name === "AbortError") {
            return `API 逾時（超過 ${config.requestTimeoutSeconds} 秒）。`;
        }
        const detail = error && error.message ? `：${error.message}` : "";
        if (window.location.protocol === "file:") {
            return `無法讀取 API；直接開啟模式需 API 允許 CORS（Access-Control-Allow-Origin: *），並請確認網路與服務狀態${detail}`;
        }
        return `無法讀取 API，請檢查網路、CORS 或服務狀態${detail}`;
    }

    function showConfigError(message) {
        const element = document.getElementById("config-error");
        element.hidden = false;
        element.textContent = message;
    }

    function initializeTheme() {
        const button = document.getElementById("theme-toggle");
        const initialTheme = document.documentElement.dataset.theme === "dark"
            ? "dark"
            : "light";
        applyTheme(initialTheme);

        button.addEventListener("click", function () {
            const nextTheme = document.documentElement.dataset.theme === "dark"
                ? "light"
                : "dark";
            try {
                localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
            } catch (_error) {
                // Theme still applies for the current page when storage is unavailable.
            }
            applyTheme(nextTheme);
        });

        if (window.matchMedia) {
            const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
            if (typeof mediaQuery.addEventListener === "function") {
                mediaQuery.addEventListener("change", function (event) {
                    if (!readStoredTheme()) {
                        applyTheme(event.matches ? "dark" : "light");
                    }
                });
            }
        }
    }

    function readStoredTheme() {
        try {
            const value = localStorage.getItem(THEME_STORAGE_KEY);
            return value === "dark" || value === "light" ? value : "";
        } catch (_error) {
            return "";
        }
    }

    function applyTheme(theme) {
        const normalizedTheme = theme === "dark" ? "dark" : "light";
        const isDark = normalizedTheme === "dark";
        document.documentElement.dataset.theme = normalizedTheme;

        const button = document.getElementById("theme-toggle");
        button.setAttribute("aria-pressed", String(isDark));
        button.setAttribute(
            "aria-label",
            isDark ? "切換為淺色模式" : "切換為深色模式"
        );
        button.title = isDark ? "切換為淺色模式" : "切換為深色模式";
        setText(button.querySelector("[data-theme-label]"), isDark ? "淺色" : "深色");

        const themeColor = document.getElementById("theme-color");
        if (themeColor) {
            themeColor.content = isDark ? "#08111c" : "#eef3f7";
        }
    }

    function setField(root, name, value) {
        setText(root.querySelector(`[data-field="${name}"]`), value);
    }

    function setText(element, value) {
        if (element) {
            element.textContent = value === undefined || value === null ? "" : String(value);
        }
    }

    function deriveBaseUrl(url) {
        try {
            const parsed = new URL(url);
            return `${parsed.protocol}//${parsed.host}/`;
        } catch (_error) {
            return "";
        }
    }

    function extractHostname(url) {
        try {
            return new URL(url).hostname;
        } catch (_error) {
            return "";
        }
    }
})();
