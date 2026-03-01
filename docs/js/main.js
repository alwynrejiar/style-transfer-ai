/* ── CUSTOM CURSOR ── */
(function initCursor() {
    const cur = document.getElementById('cur');
    const ring = document.getElementById('curRing');
    if (!cur || !ring) return;

    // Start off screen
    let mx = -100, my = -100, cx = -100, cy = -100, rx = -100, ry = -100;

    function updateMouse(e) {
        mx = e.clientX;
        my = e.clientY;
    }

    document.addEventListener('mousemove', updateMouse);
    document.addEventListener('pointermove', updateMouse);

    (function draw() {
        cx += (mx - cx) * 0.18;
        cy += (my - cy) * 0.18;
        rx += (mx - rx) * 0.09;
        ry += (my - ry) * 0.09;
        cur.style.left = cx + 'px';
        cur.style.top = cy + 'px';
        ring.style.left = rx + 'px';
        ring.style.top = ry + 'px';
        requestAnimationFrame(draw);
    })();

    document.querySelectorAll('a,button,.btn,.play-btn,.team-card,.feat,.step-node').forEach(function (el) {
        el.addEventListener('mouseenter', function () { document.body.classList.add('hov'); });
        el.addEventListener('mouseleave', function () { document.body.classList.remove('hov'); });
    });
})();

/* ── FLOATING PARTICLES ── */
(function initParticles() {
    var canvas = document.getElementById('heroParticles');
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    var home = document.getElementById('home');
    var particles = [];
    var count = 50;

    function resize() {
        var w = home.clientWidth;
        var h = home.clientHeight;
        canvas.width = w;
        canvas.height = h;
        canvas.style.width = w + 'px';
        canvas.style.height = h + 'px';
    }
    // delay first resize to ensure layout is settled
    setTimeout(resize, 100);
    window.addEventListener('resize', resize);

    function initParticlePositions() {
        particles = [];
        for (var i = 0; i < count; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                r: Math.random() * 2.5 + 0.5,
                dx: (Math.random() - 0.5) * 0.5,
                dy: (Math.random() - 0.5) * 0.5,
                opacity: Math.random() * 0.15 + 0.03
            });
        }
    }
    initParticlePositions();

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        for (var i = 0; i < particles.length; i++) {
            var p = particles[i];
            p.x += p.dx;
            p.y += p.dy;
            if (p.x < 0 || p.x > canvas.width) p.dx *= -1;
            if (p.y < 0 || p.y > canvas.height) p.dy *= -1;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(26,25,22,' + p.opacity + ')';
            ctx.fill();
        }
        // connect nearby particles with lines
        for (var i = 0; i < particles.length; i++) {
            for (var j = i + 1; j < particles.length; j++) {
                var dx = particles[i].x - particles[j].x;
                var dy = particles[i].y - particles[j].y;
                var dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 120) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = 'rgba(160,158,153,' + (0.08 * (1 - dist / 120)) + ')';
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(draw);
    }
    draw();
})();

/* ── PARALLAX BG CHARS ON MOUSE ── */
(function initParallax() {
    var char1 = document.querySelector('.bg-char-1');
    var char2 = document.querySelector('.bg-char-2');
    if (!char1 || !char2) return;

    document.getElementById('home').addEventListener('mousemove', function (e) {
        var rect = e.currentTarget.getBoundingClientRect();
        var x = (e.clientX - rect.left) / rect.width - 0.5;
        var y = (e.clientY - rect.top) / rect.height - 0.5;
        char1.style.transform = 'translate(' + (x * -20) + 'px,' + (y * -15) + 'px)';
        char2.style.transform = 'translate(' + (x * 15) + 'px,' + (y * 12) + 'px)';
    });
})();

/* ── WORD-BY-WORD REVEAL ── */
(function initWordReveal() {
    var desc = document.querySelector('.hero-desc');
    if (!desc) return;
    var html = desc.innerHTML;
    // wrap each text word in a span, preserving HTML tags
    var result = '';
    var inTag = false;
    var wordIndex = 0;
    var buffer = '';

    function flushWord() {
        if (buffer.trim()) {
            result += '<span class="word" style="transition-delay:' + (wordIndex * 0.04) + 's">' + buffer + '</span> ';
            wordIndex++;
        } else if (buffer) {
            result += buffer;
        }
        buffer = '';
    }

    for (var i = 0; i < html.length; i++) {
        var ch = html[i];
        if (ch === '<') {
            flushWord();
            inTag = true;
            buffer += ch;
        } else if (ch === '>') {
            buffer += ch;
            result += buffer;
            buffer = '';
            inTag = false;
        } else if (inTag) {
            buffer += ch;
        } else if (ch === ' ' || ch === '\n') {
            flushWord();
        } else {
            buffer += ch;
        }
    }
    flushWord();

    desc.innerHTML = result;

    // trigger after a brief delay
    setTimeout(function () {
        var words = desc.querySelectorAll('.word');
        words.forEach(function (w) {
            w.classList.add('visible');
        });
    }, 700);
})();

/* ── SCROLL ANIMATIONS ── */
(function initScrollAnimations() {
    document.documentElement.setAttribute('data-anim-ready', '');

    var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
            if (e.isIntersecting) {
                e.target.classList.add('shown');
            }
        });
    }, { threshold: 0.15 });

    document.querySelectorAll('[data-anim]').forEach(function (el) {
        observer.observe(el);
    });
})();

/* ── ACTIVE NAV LINK ── */
(function initActiveNav() {
    var sections = document.querySelectorAll('section[id]');
    var links = document.querySelectorAll('.nav-links a');

    window.addEventListener('scroll', function () {
        var scrollY = window.scrollY + 120;
        sections.forEach(function (sec) {
            if (scrollY >= sec.offsetTop && scrollY < sec.offsetTop + sec.offsetHeight) {
                links.forEach(function (l) { l.classList.remove('active'); });
                var active = document.querySelector('.nav-links a[href="#' + sec.id + '"]');
                if (active) active.classList.add('active');
            }
        });
    });
})();

/* ── ROADMAP STEP ANIMATIONS ── */
(function initRoadmap() {
    var steps = document.querySelectorAll('.step');
    if (steps.length === 0) return;

    var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
            if (entry.isIntersecting) {
                var allSteps = document.querySelectorAll('.step');
                allSteps.forEach(function (step, i) {
                    setTimeout(function () {
                        step.classList.add('step-visible');
                    }, i * 150);
                });
                observer.disconnect();
            }
        });
    }, { threshold: 0.2 });

    observer.observe(document.getElementById('roadmap'));
})();

/* ── FEATURE CARD REVEAL ── */
(function initFeatureCards() {
    var feats = document.querySelectorAll('.feat');
    if (feats.length === 0) return;

    var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
            if (entry.isIntersecting) {
                entry.target.classList.add('feat-visible');
            }
        });
    }, { threshold: 0.12 });

    feats.forEach(function (f) { observer.observe(f); });
})();

/* ── MOBILE HAMBURGER MENU ── */
(function initHamburger() {
    var hamburger = document.getElementById('hamburger');
    var navLinks = document.getElementById('navLinks');
    if (!hamburger || !navLinks) return;

    function openMenu() {
        hamburger.classList.add('active');
        navLinks.classList.add('open');
        document.body.classList.add('menu-open');
    }

    function closeMenu() {
        hamburger.classList.remove('active');
        navLinks.classList.remove('open');
        document.body.classList.remove('menu-open');
    }

    hamburger.addEventListener('click', function () {
        if (navLinks.classList.contains('open')) {
            closeMenu();
        } else {
            openMenu();
        }
    });

    navLinks.querySelectorAll('a').forEach(function (link) {
        link.addEventListener('click', function (e) {
            e.preventDefault();
            var href = this.getAttribute('href');
            closeMenu();
            var target = document.querySelector(href);
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
            // update active state
            navLinks.querySelectorAll('a').forEach(function (l) { l.classList.remove('active'); });
            this.classList.add('active');
        });
    });
})();
