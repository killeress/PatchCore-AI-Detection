(() => {
    'use strict';
    document.addEventListener('click', (event) => {
        const trigger = event.target.closest('button[data-edge-evidence]');
        if (!trigger) return;
        const panel = document.getElementById(trigger.dataset.edgeEvidence);
        if (!panel) return;
        const open = panel.hidden;
        panel.hidden = !open;
        document.querySelectorAll('button[data-edge-evidence]').forEach((button) => {
            if (button.dataset.edgeEvidence !== panel.id) return;
            button.setAttribute('aria-expanded', String(open));
            button.textContent = open ? '收起依據 ▴' : '查看依據 ▾';
        });
    });
})();
