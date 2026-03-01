/* ── CUSTOM CURSOR ── */
(function initCursor() {
    const cur = document.getElementById('cur');
    const ring = document.getElementById('curRing');
    if (!cur || !ring) return;

    let mx = 0, my = 0, cx = 0, cy = 0, rx = 0, ry = 0;

    window.addEventListener('mousemove', function (e) {
        mx = e.clientX;
        my = e.clientY;
    });

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

    var scrollPos = 0;

    function openMenu() {
        scrollPos = window.scrollY;
        hamburger.classList.add('active');
        navLinks.classList.add('open');
        document.body.classList.add('menu-open');
        document.body.style.top = -scrollPos + 'px';
    }

    function closeMenu() {
        hamburger.classList.remove('active');
        navLinks.classList.remove('open');
        document.body.classList.remove('menu-open');
        document.body.style.top = '';
        window.scrollTo(0, scrollPos);
    }

    hamburger.addEventListener('click', function () {
        if (navLinks.classList.contains('open')) {
            closeMenu();
        } else {
            openMenu();
        }
    });

    navLinks.querySelectorAll('a').forEach(function (link) {
        link.addEventListener('click', function () {
            closeMenu();
        });
    });
})();
