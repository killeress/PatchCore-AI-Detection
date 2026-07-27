(function () {
    "use strict";

    const DEFAULT_REFRESH_SECONDS = 30;
    const MIN_REFRESH_SECONDS = 30;
    const DEFAULT_TIMEOUT_SECONDS = 8;
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
        const factoryGrids = new Map();
        for (const line of activeLines) {
            if (!line.id || seenIds.has(line.id)) {
                showConfigError("每條線都必須有不重複的 id，請檢查 config.js。");
                continue;
            }
            seenIds.add(line.id);
            const factory = line.factory || "未設定廠別";
            if (!factoryGrids.has(factory)) {
                factoryGrids.set(factory, createFactorySection(factory, factoryGrids.size + 1));
            }
            createLineCard(line, factoryGrids.get(factory));
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

        card.dataset.lineId = line.id;
        setField(card, "pc-name", line.pcName || line.id);
        setField(card, "line-name", line.line || "未設定線體");

        configureLink(card, "dashboard", line.dashboardUrl || deriveBaseUrl(line.apiUrl));
        configureLink(card, "overexposed", line.overexposedUrl);

        lineGrid.appendChild(card);
        lineStates.set(line.id, {
            line,
            card,
            status: "checking",
            data: null,
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
                    message: `硬碟空間嚴重不足：剩餘 ${formatDecimal(freePercent, 1)}%（${formatDecimal(hardware.diskFreeGb, 1)} / ${formatDecimal(hardware.diskTotalGb, 1)} GB）`
                });
            } else if (freePercent <= HEALTH_THRESHOLDS.diskFreeWarningPercent) {
                alerts.push({
                    severity: "warning",
                    message: `硬碟空間偏低：剩餘 ${formatDecimal(freePercent, 1)}%（${formatDecimal(hardware.diskFreeGb, 1)} / ${formatDecimal(hardware.diskTotalGb, 1)} GB）`
                });
            }
        }

        if (hardware.ramUsedPercent !== null) {
            if (hardware.ramUsedPercent >= HEALTH_THRESHOLDS.ramUsedCriticalPercent) {
                alerts.push({
                    severity: "critical",
                    message: `RAM 使用率過高：${formatDecimal(hardware.ramUsedPercent, 0)}%`
                });
            } else if (hardware.ramUsedPercent >= HEALTH_THRESHOLDS.ramUsedWarningPercent) {
                alerts.push({
                    severity: "warning",
                    message: `RAM 使用率偏高：${formatDecimal(hardware.ramUsedPercent, 0)}%`
                });
            }
        }

        if (hardware.vramUsedGb !== null && hardware.vramTotalGb > 0) {
            const usedPercent = (hardware.vramUsedGb / hardware.vramTotalGb) * 100;
            if (usedPercent >= HEALTH_THRESHOLDS.vramUsedCriticalPercent) {
                alerts.push({
                    severity: "critical",
                    message: `VRAM 使用率過高：${formatDecimal(usedPercent, 0)}%（${formatDecimal(hardware.vramUsedGb, 1)} / ${formatDecimal(hardware.vramTotalGb, 1)} GB）`
                });
            } else if (usedPercent >= HEALTH_THRESHOLDS.vramUsedWarningPercent) {
                alerts.push({
                    severity: "warning",
                    message: `VRAM 使用率偏高：${formatDecimal(usedPercent, 0)}%（${formatDecimal(hardware.vramUsedGb, 1)} / ${formatDecimal(hardware.vramTotalGb, 1)} GB）`
                });
            }
        }

        if (hardware.gpuTemperature !== null) {
            if (hardware.gpuTemperature >= HEALTH_THRESHOLDS.gpuTemperatureCriticalC) {
                alerts.push({
                    severity: "critical",
                    message: `GPU 溫度過高：${formatDecimal(hardware.gpuTemperature, 0)}°C`
                });
            } else if (hardware.gpuTemperature >= HEALTH_THRESHOLDS.gpuTemperatureWarningC) {
                alerts.push({
                    severity: "warning",
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

    function deriveBaseUrl(url) {
        try {
            const parsed = new URL(url);
            return `${parsed.protocol}//${parsed.host}/`;
        } catch (_error) {
            return "";
        }
    }
})();
