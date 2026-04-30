(function () {
  var form = document.getElementById('risk-formu');
  if (!form) {
    return;
  }

  form.addEventListener('submit', function () {
    var buton = form.querySelector('button[type="submit"]');
    if (!buton) {
      return;
    }
    buton.disabled = true;
    buton.textContent = 'Hesaplanıyor...';
  });
})();
