/* Shared by record details, the AJAX report and DEBUG results. */
(() => {
    'use strict';
    let dialog, image, canvas, stage, status, marker, markerButton, opener;
    let fit = true, showMarker = true, frameRatio = 0;

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
        const trigger = event.target.closest('a[data-sw-image]');
        if (!trigger) return;
        event.preventDefault();
        if (trigger.classList.contains('sw-image-missing')) return;
        if (!dialog) createViewer();
        opener = trigger;
        fit = true;
        showMarker = true;
        frameRatio = Number(trigger.dataset.swFrameRatio) || 0;
        markerButton.hidden = !frameRatio;
        markerButton.setAttribute('aria-pressed', 'true');
        dialog.querySelector('[data-sw-caption]').textContent = trigger.dataset.swTitle || '影像';
        image.alt = trigger.dataset.swTitle || '影像';
        canvas.hidden = true;
        status.hidden = false;
        status.textContent = '載入中…';
        dialog.showModal();
        stage.scrollTop = stage.scrollLeft = 0;
        image.src = trigger.href;
    });
})();
