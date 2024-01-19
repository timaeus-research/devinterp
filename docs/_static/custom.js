window.addEventListener('load', function() {
    var headers = document.querySelectorAll('div.body h1');
    headers.forEach(function(header) {
        if (header.textContent == 'Module contents') {
            header.style.display = 'none';
        }
    });
});
