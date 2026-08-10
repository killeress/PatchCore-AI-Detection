(function () {
    "use strict";

    const DEFAULT_REFRESH_SECONDS = 30;
    const MIN_REFRESH_SECONDS = 30;
    const DEFAULT_TIMEOUT_SECONDS = 8;
    const THEME_STORAGE_KEY = "capi-dashboard-theme";
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

    let refreshTimer = null;
    let countdownTimer = null;
    let clockTimer = null;
    let nextRefreshAt = null;
    let isRefreshing = false;

    document.addEventListener("DOMContentLoaded", initialize);

    async function initialize() {
        initializeTheme();
        const directFileMode = window.location.protocol === "file:";
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

        const activeLines = config.lines.filter((line) => line.enabled !== false);
        if (activeLines.length === 0) {
            showConfigError("尚未設定任何啟用中的線體。");
            updateSummary();
            startClock();
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
        updateSummary();

        startClock();
        countdownTimer = window.setInterval(updateRefreshStatus, 1000);
        refreshAllLines();
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
            lines: Array.isArray(raw.lines) ? raw.lines : []
        };
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

        card.dataset.lineId = line.id;
        setField(card, "pc-name", line.pcName || line.id);
        setField(card, "line-name", line.line || "未設定線體");

        configureLink(card, "dashboard", line.dashboardUrl || deriveBaseUrl(line.apiUrl));

        lineGrid.appendChild(card);
        lineStates.set(line.id, {
            line,
            card,
            overviewRow,
            status: "checking",
            data: null,
            error: ""
        });
        renderLineCard(lineStates.get(line.id));
    }

    function createOverviewRow(line) {
        const row = document.createElement("tr");
        row.dataset.lineId = line.id;
        row.dataset.state = "checking";

        const lineCell = document.createElement("td");
        lineCell.className = "overview-line-cell";
        const lineIdentity = document.createElement("div");
        lineIdentity.className = "overview-line";
        const factory = document.createElement("span");
        factory.className = "overview-factory";
        factory.textContent = line.factory || "未設定廠別";
        const lineName = document.createElement("strong");
        lineName.textContent = line.line || "未設定線體";
        lineIdentity.append(factory, lineName);
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
            activityCell,
            alertCell,
            linkCell
        );
        configureLink(row, "dashboard", line.dashboardUrl || deriveBaseUrl(line.apiUrl));
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
        if (isRefreshing || lineStates.size === 0) {
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
            state.status = state.data.running ? "online" : "warning";
            state.error = state.data.running ? "" : "API 可連線，但服務回報未運行。";
        } catch (error) {
            state.status = "offline";
            state.error = readableFetchError(error);
        } finally {
            clearTimeout(timeoutId);
            renderLineCard(state);
        }
    }

    function normalizeStatus(raw) {
        const server = asObject(raw.server);
        const traffic = asObject(raw.traffic);
        const stats = asObject(raw.stats);
        const latestEvent = asObject(raw.latest_event);
        const hardware = asObject(raw.hardware || server.hardware);
        const gpu = asObject(hardware.gpu);
        const memory = asObject(hardware.memory || hardware.ram);
        const disk = asObject(hardware.disk);

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
                machineNo: textValue(latestEvent.machine_no),
                judgment: textValue(latestEvent.judgment || latestEvent.detail),
                time: textValue(latestEvent.time),
                duration: textValue(latestEvent.duration)
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
            }
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
        setField(card, "gpu-health", formatGpuHealth(data.hardware));
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

        renderOverviewActivity(state);

        const alertCell = row.querySelector(".overview-alert-cell");
        const alertContainer = row.querySelector('[data-field="overview-alerts"]');
        const healthAlerts = data && state.status !== "offline"
            ? getHardwareAlerts(data)
            : [];
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
        const states = Array.from(lineStates.values());

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
    }

    function renderAlerts() {
        const alerts = [];
        for (const state of lineStates.values()) {
            const label = `${state.line.factory || "未設定廠別"} / ${state.line.line || state.line.id}`;
            if (state.status === "offline") {
                alerts.push({
                    severity: "critical",
                    message: `${label}：${state.error || statusText(state.status)}`
                });
            } else if (state.status === "warning") {
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
