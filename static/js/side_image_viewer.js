/* Shared by record details, the AJAX report and DEBUG results. */
(() => {
    'use strict';
    let dialog, image, canvas, stage, status, marker, markerButton, candidatesButton, opener;
    let fit = true, showMarker = true, showCandidates = true, frameRatio = 0;
    let cleanUrl = '', markedUrl = '';

    function loadImage(url) {
        canvas.hidden = true;
        status.hidden = false;
        status.textContent = '載入中…';
        image.src = url;
    }

    function layout() {
        if (!dialog.open || !image.naturalWidth || canvas.hidden) return;
        const scale = fit ? Math.min((stage.clientWidth - 32) / image.naturalWidth,
            (stage.clientHeight - 32) / image.naturalHeight) : 1;
        canvas.style.width = `${Math.max(1, image.naturalWidth * scale)}px`;
        canvas.style.height = `${Math.max(1, image.naturalHeight * scale)}px`;
        marker.style.setProperty('--sw-frame-size', `${frameRatio * 100}%`);
        marker.hidden = !frameRatio || !showMarker;
        dialog.querySelector('[data-sw-fit]').setAttribute('aria-pressed', String(fit));
        dialog.querySelector('[data-sw-native]').setAttribute('aria-pressed', String(!fit));
    }

    function createViewer() {
        dialog = document.createElement('dialog');
        dialog.className = 'sw-viewer';
        dialog.setAttribute('aria-label', '影像放大檢視');
        dialog.innerHTML = `<div class="sw-viewer-bar"><strong data-sw-caption></strong>
            <button type="button" data-sw-fit aria-pressed="true">適合視窗</button>
            <button type="button" data-sw-native aria-pressed="false">1:1 原始像素</button>
            <button type="button" data-sw-marker aria-pressed="true">顯示 AOI 框</button>
            <button type="button" data-sw-candidates aria-pressed="true">顯示候選與距離</button>
            <button type="button" data-sw-close autofocus aria-label="關閉放大檢視">關閉 ×</button></div>
            <div class="sw-viewer-stage"><p class="sw-viewer-status" role="status">載入中…</p>
            <div class="sw-viewer-canvas" hidden><img alt=""><span class="sw-aoi-frame" aria-hidden="true"></span></div></div>
            <p class="sw-viewer-note">1:1 可捲動查看原始像素；Esc 或點擊背景關閉。</p>`;
        document.body.append(dialog);
        image = dialog.querySelector('img');
        canvas = dialog.querySelector('.sw-viewer-canvas');
        stage = dialog.querySelector('.sw-viewer-stage');
        status = dialog.querySelector('[role=status]');
        marker = dialog.querySelector('.sw-aoi-frame');
        markerButton = dialog.querySelector('[data-sw-marker]');
        candidatesButton = dialog.querySelector('[data-sw-candidates]');
        image.addEventListener('load', () => {
            status.hidden = true;
            canvas.hidden = false;
            layout();
        });
        image.addEventListener('error', () => {
            canvas.hidden = true;
            status.hidden = false;
            status.textContent = '影像已過期或遺失，請重新查詢。';
        });
        dialog.addEventListener('click', event => {
            if (event.target === dialog || event.target.closest('[data-sw-close]')) dialog.close();
            else if (event.target.closest('[data-sw-fit]')) { fit = true; layout(); }
            else if (event.target.closest('[data-sw-native]')) { fit = false; layout(); }
            else if (event.target.closest('[data-sw-marker]')) {
                showMarker = !showMarker;
                markerButton.setAttribute('aria-pressed', String(showMarker));
                layout();
            } else if (event.target.closest('[data-sw-candidates]')) {
                showCandidates = !showCandidates;
                candidatesButton.setAttribute('aria-pressed', String(showCandidates));
                loadImage(showCandidates ? markedUrl : cleanUrl);
            }
        });
        dialog.addEventListener('close', () => {
            image.removeAttribute('src');
            canvas.hidden = true;
            if (opener?.isConnected) opener.focus();
        });
        window.addEventListener('resize', layout);
    }

    document.addEventListener('click', event => {
        const toggle = event.target.closest('[data-sw-candidates-toggle]');
        if (toggle) {
            event.preventDefault();
            const show = toggle.getAttribute('aria-pressed') !== 'true';
            toggle.setAttribute('aria-pressed', String(show));
            toggle.textContent = show ? '隱藏候選與距離' : '顯示候選與距離';
            toggle.closest('[data-sw-aoi-point]').querySelectorAll('a[data-sw-marked-url]').forEach(link => {
                if (link.classList.contains('sw-image-missing')) return;
                const url = show ? link.dataset.swMarkedUrl : link.dataset.swCleanUrl;
                link.dataset.swCandidatesVisible = String(show);
                link.href = url;
                link.querySelector('img').src = url;
            });
            return;
        }
        const trigger = event.target.closest('a[data-sw-image]');
        if (!trigger) return;
        event.preventDefault();
        if (trigger.classList.contains('sw-image-missing')) return;
        if (!dialog) createViewer();
        opener = trigger;
        fit = true;
        showMarker = true;
        showCandidates = trigger.dataset.swCandidatesVisible !== 'false';
        cleanUrl = trigger.dataset.swCleanUrl || '';
        markedUrl = trigger.dataset.swMarkedUrl || '';
        frameRatio = Number(trigger.dataset.swFrameRatio) || 0;
        markerButton.hidden = !frameRatio;
        markerButton.setAttribute('aria-pressed', 'true');
        candidatesButton.hidden = !cleanUrl || !markedUrl;
        candidatesButton.setAttribute('aria-pressed', String(showCandidates));
        dialog.querySelector('[data-sw-caption]').textContent = trigger.dataset.swTitle || '影像';
        image.alt = trigger.dataset.swTitle || '影像';
        dialog.showModal();
        stage.scrollTop = stage.scrollLeft = 0;
        loadImage(trigger.href);
    });
})();

/* Delegation also covers side-statistics and DEBUG cards inserted after page load. */
(() => {
    'use strict';

    async function copyText(text) {
        if (navigator.clipboard?.writeText) {
            try { await navigator.clipboard.writeText(text); return; }
            catch (_) { /* HTTP/restricted clipboard: use the local selection fallback. */ }
        }
        const previous = document.activeElement;
        const area = document.createElement('textarea');
        area.value = text;
        area.readOnly = true;
        area.style.cssText = 'position:fixed;left:-9999px;top:0';
        document.body.append(area);
        try {
            area.focus({preventScroll: true});
            area.select();
            if (!document.execCommand('copy')) throw new Error('clipboard');
        } finally {
            area.remove();
            previous?.focus({preventScroll: true});
        }
    }

    async function copyDiagnostics(button) {
        if (button.disabled) return;
        const card = button.closest('section');
        const status = card.querySelector('[data-sw-copy-status]');
        const originalLabel = button.textContent;
        let text;
        button.disabled = true;
        button.setAttribute('aria-busy', 'true');
        button.textContent = '複製中…';
        status.hidden = false;
        status.textContent = '正在準備側拍 LOG 與完整 payload…';
        card.querySelector('[data-sw-copy-manual]')?.remove();
        try {
            let data;
            const inline = card.querySelector('[data-sw-debug-diagnostics]');
            if (inline) data = JSON.parse(inline.textContent);
            else {
                const id = button.dataset.swResultId || '';
                if (!/^[1-9]\d*$/.test(id)) throw new Error('沒有有效的側拍結果 ID，請重新查詢');
                const response = await fetch('/api/side-white/diagnostics?id=' + encodeURIComponent(id),
                    {cache: 'no-store', headers: {'Accept': 'application/json'}});
                data = await response.json();
                if (!response.ok || data.error) throw new Error(data.error || '側拍排查資料讀取失敗');
            }
            if (!data.payload || typeof data.payload !== 'object' || typeof data.side_log !== 'string') {
                throw new Error('排查資料格式不完整，請重新查詢');
            }
            const {payload, side_log, ...context} = data;
            text = '=== 側拍排查資料 ===\n' + JSON.stringify({
                ...context, copied_at: new Date().toISOString(), page_url: window.location.href,
            }, null, 2) + '\n\n=== 側拍 LOG ===\n' + side_log +
                '\n\n=== 側拍結果 payload（完整 JSON） ===\n' + JSON.stringify(payload, null, 2);
            await copyText(text);
            status.textContent = '已複製側拍 LOG、完整 payload 與紀錄資訊，可直接貼上排查。';
        } catch (error) {
            if (text !== undefined) {
                const area = document.createElement('textarea');
                area.className = 'sw-copy-manual';
                area.setAttribute('data-sw-copy-manual', '');
                area.setAttribute('aria-label', '側拍排查資料，請手動複製');
                area.readOnly = true;
                area.value = text;
                status.after(area);
                area.focus();
                area.select();
                status.textContent = '瀏覽器無法自動複製，資料已選取，請按 Ctrl+C 複製。';
            } else status.textContent = '複製失敗：' + (error.message || String(error));
        } finally {
            button.disabled = false;
            button.removeAttribute('aria-busy');
            button.textContent = originalLabel;
            if (document.activeElement === document.body) button.focus({preventScroll: true});
        }
    }

    document.addEventListener('click', event => {
        const button = event.target.closest('[data-sw-copy-diagnostics]');
        if (!button) return;
        event.preventDefault();
        void copyDiagnostics(button);
    });
})();
