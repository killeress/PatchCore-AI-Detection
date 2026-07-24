(function () {
    "use strict";

    const DEFAULT_REFRESH_SECONDS = 30;
    const MIN_REFRESH_SECONDS = 30;
    const DEFAULT_TIMEOUT_SECONDS = 8;
    const config = normalizeConfig(window.CAPI_DASHBOARD_CONFIG);
    const lineStates = new Map();

    let refreshTimer = null;
    let countdownTimer = null;
    let clockTimer = null;
    let nextRefreshAt = null;
    let isRefreshing = false;

    document.addEventListener("DOMContentLoaded", initialize);

    function initialize() {
        document.title = config.title;
        setText(document.getElementById("dashboard-title"), config.title);
        const directFileMode = window.location.protocol === "file:";
        setText(
            document.getElementById("data-note"),
            `${directFileMode ? "直接開啟模式 · " : ""}每 ${config.refreshIntervalSeconds} 秒由各 PC 的 API 更新一次`
        );
        setText(
            document.getElementById("footer-refresh-note"),
            `更新週期：${config.refreshIntervalSeconds} 秒`
        );

        const activeLines = config.lines.filter((line) => line.enabled !== false);
        if (activeLines.length === 0) {
            showConfigError("config.js 尚未設定任何啟用中的線體。");
            updateSummary();
            startClock();
            return;
        }

        const seenIds = new Set();
        for (const line of activeLines) {
            if (!line.id || seenIds.has(line.id)) {
                showConfigError("每條線都必須有不重複的 id，請檢查 config.js。");
                continue;
            }
            seenIds.add(line.id);
            createLineCard(line);
        }

        document.getElementById("refresh-button").addEventListener("click", function () {
            clearTimeout(refreshTimer);
            refreshAllLines();
        });

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
            title: String(raw.title || "CAPI AI 中控看板"),
            refreshIntervalSeconds,
            requestTimeoutSeconds,
            lines: Array.isArray(raw.lines) ? raw.lines : []
        };
    }

    function toPositiveInteger(value, fallback) {
        const number = Number(value);
        return Number.isFinite(number) && number > 0 ? Math.round(number) : fallback;
    }

    function createLineCard(line) {
        const template = document.getElementById("line-card-template");
        const card = template.content.firstElementChild.cloneNode(true);

        card.dataset.lineId = line.id;
        setField(card, "factory", line.factory || "未設定廠別");
        setField(card, "pc-name", line.pcName || line.id);
        setField(card, "line-name", line.line || "未設定線體");
        setField(card, "api-host", safeUrlLabel(line.apiUrl));

        configureLink(card, "dashboard", line.dashboardUrl || deriveBaseUrl(line.apiUrl));
        configureLink(card, "overexposed", line.overexposedUrl);

        document.getElementById("line-grid").appendChild(card);
        lineStates.set(line.id, {
            line,
            card,
            status: "checking",
            data: null,
            lastSuccessAt: null,
            lastAttemptAt: null,
            error: ""
        });
        renderLineCard(lineStates.get(line.id));
    }

    function configureLink(card, name, url) {
        const element = card.querySelector(`[data-link="${name}"]`);
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
        const button = document.getElementById("refresh-button");
        button.disabled = true;
        button.textContent = "更新中…";
        updateRefreshStatus();

        await Promise.all(Array.from(lineStates.values(), refreshLine));

        isRefreshing = false;
        button.disabled = false;
        button.textContent = "立即更新";
        updateSummary();
        renderAlerts();
        scheduleNextRefresh();
    }

    async function refreshLine(state) {
        state.lastAttemptAt = new Date();
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
            state.lastSuccessAt = new Date();
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
            uptime: textValue(server.uptime),
            modelVersion: textValue(server.model_version),
            device: textValue(server.device || gpu.name),
            total: numberValue(stats.total_requests ?? stats.total),
            ok: numberValue(stats.total_ok ?? stats.ok_count),
            ng: numberValue(stats.total_ng ?? stats.ng_count),
            err: numberValue(stats.total_err ?? stats.err_count),
            shiftName: textValue(stats.shift_name),
            shiftRange: textValue(stats.time_range),
            overexposed: optionalNumber(stats.overexposed_count),
            avgTime: optionalNumber(
                stats.avg_time ??
                stats.average_processing_seconds ??
                asObject(raw.performance).avg_seconds
            ),
            activeConnections: numberValue(traffic.active_connections),
            activeInferences: numberValue(traffic.active_inferences),
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
        card.dataset.state = state.status;
        setField(card, "status", statusText(state.status));
        setField(card, "last-success", formatFreshness(state.lastSuccessAt));

        const errorElement = card.querySelector('[data-field="error"]');
        errorElement.hidden = !state.error;
        setText(errorElement, state.error);

        if (!data) {
            return;
        }

        const validCount = data.ok + data.ng;
        const totalForErrorRate = data.total || validCount + data.err;

        setField(card, "shift-name", data.shiftName || "當班");
        setField(card, "shift-range", data.shiftRange || "API 未提供班別時段");
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
        setField(card, "ram", formatPercent(data.hardware.ramUsedPercent));
        setField(card, "disk", formatDisk(data.hardware));
        setField(
            card,
            "connections",
            `${formatNumber(data.activeConnections)} / ${formatNumber(data.connectedMachines.length)} 機台`
        );
        setField(card, "active-inferences", formatNumber(data.activeInferences));

        const latestLabel = [data.latestEvent.glassId, data.latestEvent.machineNo]
            .filter(Boolean)
            .join(" / ");
        setField(card, "latest-glass", latestLabel || "尚無資料");
        setField(card, "latest-judgment", displayJudgment(data.latestEvent.judgment));
        setField(card, "latest-time", data.latestEvent.time || "—");
        updateJudgmentClass(card, data.latestEvent.judgment);
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

    function formatPercent(value) {
        return value === null ? "—" : `${formatDecimal(value, 0)}%`;
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

    function updateSummary() {
        const states = Array.from(lineStates.values());
        const onlineStates = states.filter((state) => state.status === "online" && state.data);
        const aggregate = onlineStates.reduce(
            (result, state) => {
                result.total += state.data.total;
                result.ok += state.data.ok;
                result.ng += state.data.ng;
                return result;
            },
            { total: 0, ok: 0, ng: 0 }
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
        setText(document.getElementById("summary-shift-total"), formatNumber(aggregate.total));
        setText(
            document.getElementById("summary-ng-rate"),
            formatRate(aggregate.ng, aggregate.ok + aggregate.ng)
        );
    }

    function renderAlerts() {
        const alerts = Array.from(lineStates.values()).filter(
            (state) => state.status === "offline" || state.status === "warning"
        );
        const panel = document.getElementById("alert-panel");
        const list = document.getElementById("alert-list");
        list.replaceChildren();

        if (alerts.length === 0) {
            panel.hidden = true;
            return;
        }

        for (const state of alerts) {
            const item = document.createElement("li");
            const label = `${state.line.factory || "未設定廠別"} / ${state.line.line || state.line.id}`;
            item.textContent = `${label}：${state.error || statusText(state.status)}`;
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
            updateFreshnessLabels();
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
    }

    function updateRefreshStatus() {
        const element = document.getElementById("refresh-status");
        if (isRefreshing) {
            setText(element, "正在更新各 PC");
            return;
        }
        if (!nextRefreshAt) {
            setText(element, "準備更新");
            return;
        }
        const seconds = Math.max(0, Math.ceil((nextRefreshAt - Date.now()) / 1000));
        setText(element, `${seconds} 秒後更新`);
    }

    function updateFreshnessLabels() {
        for (const state of lineStates.values()) {
            setField(state.card, "last-success", formatFreshness(state.lastSuccessAt));
        }
    }

    function formatFreshness(date) {
        if (!date) {
            return "尚未取得";
        }
        const seconds = Math.max(0, Math.round((Date.now() - date.getTime()) / 1000));
        if (seconds < 5) {
            return "剛剛";
        }
        if (seconds < 60) {
            return `${seconds} 秒前`;
        }
        return `${Math.floor(seconds / 60)} 分鐘前`;
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

    function setField(card, name, value) {
        setText(card.querySelector(`[data-field="${name}"]`), value);
    }

    function setText(element, value) {
        if (element) {
            element.textContent = value === undefined || value === null ? "" : String(value);
        }
    }

    function safeUrlLabel(url) {
        try {
            const parsed = new URL(url);
            return `${parsed.host}${parsed.pathname}`;
        } catch (_error) {
            return url || "未設定 API";
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
})();
