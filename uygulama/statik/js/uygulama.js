(function () {
  // Form progress + submit lock
  var form = document.getElementById('risk-formu');
  if (form) {
    var bar = document.getElementById('form-progress');
    var count = document.getElementById('form-progress-count');
    var inputs = form.querySelectorAll('input[required]');
    var total = inputs.length;

    function refresh() {
      var dolu = 0;
      inputs.forEach(function (i) {
        if (i.value && i.value.trim() !== '') dolu++;
      });
      var pct = total ? Math.round((dolu / total) * 100) : 0;
      if (bar) bar.style.setProperty('--w', pct + '%');
      if (count) count.textContent = String(dolu);
    }

    inputs.forEach(function (i) {
      i.addEventListener('input', refresh);
      i.addEventListener('change', refresh);
    });
    form.addEventListener('reset', function () { setTimeout(refresh, 0); });
    refresh();

    form.addEventListener('submit', function () {
      var btn = form.querySelector('button[type="submit"]');
      if (!btn) return;
      btn.disabled = true;
      var span = btn.querySelector('span');
      if (span) span.textContent = 'Hesaplanıyor...';
    });
  }

  // Sonuç sayfası: tüm faktörleri aç/kapa
  var toggles = document.querySelectorAll('[data-toggle]');
  toggles.forEach(function (btn) {
    var hedefId = btn.getAttribute('data-toggle');
    var hedef = document.getElementById(hedefId);
    if (!hedef) return;
    var label = btn.querySelector('.ft-text');
    var ok = btn.querySelector('.ft-arrow');

    btn.addEventListener('click', function () {
      var acik = !hedef.hasAttribute('hidden') ? false : true;
      if (acik) {
        hedef.removeAttribute('hidden');
        btn.setAttribute('aria-expanded', 'true');
        btn.classList.add('is-open');
        if (label) label.textContent = 'Faktörleri Gizle';
        if (ok) ok.textContent = '↑';
      } else {
        hedef.setAttribute('hidden', '');
        btn.setAttribute('aria-expanded', 'false');
        btn.classList.remove('is-open');
        if (label) label.textContent = 'Tüm Faktörleri Gör';
        if (ok) ok.textContent = '↓';
      }
    });
  });
})();
