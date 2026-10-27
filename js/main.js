/* Vasan · contact sheet
   Develop-in hero, sticker table, contact-sheet strip with print dialog,
   tossable prints, copy-to-clipboard email, and the easter eggs.
   Reduced-motion visitors get the same page, settled and still. */

(() => {
  "use strict";

  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const params = new URLSearchParams(location.search);
  const root = document.documentElement;
  let converted = false;

  /* ---------- modes: daylight / silver / safelight ---------- */
  if (params.has("mode")) {
    const q = params.get("mode");
    const targetMode = q === "daylight" ? "" : q;
    root.classList.remove("silver", "safe");
    if (targetMode) root.classList.add(targetMode);
  }
  let mode = root.classList.contains("silver") ? 1 : root.classList.contains("safe") ? 2 : 0;
  const updateThemeColor = () => {
    const meta = $('meta[name="theme-color"]');
    if (meta) {
      if (mode === 2) meta.content = converted ? "#03060f" : "#0c0403";
      else meta.content = mode === 1 ? "#121110" : "#f1efea";
    }
  };
  const paintModes = () => {
    $$(".modes button").forEach((b) =>
      b.setAttribute("aria-pressed", String(["", "silver", "safe"][mode] === b.dataset.mode)));
    updateThemeColor();
  };
  paintModes();
  $$(".modes button").forEach((b) => b.addEventListener("click", () => {
    root.classList.remove("silver", "safe");
    if (b.dataset.mode) root.classList.add(b.dataset.mode);
    mode = ["", "silver", "safe"].indexOf(b.dataset.mode);
    try { localStorage.setItem("vr-mode", b.dataset.mode); } catch (e) { /* private mode */ }
    if (converted) {
      const isSafe = root.classList.contains("safe");
      root.style.setProperty("--red", isSafe ? "#60a5fa" : "#2f62d6");
      root.style.setProperty("--amber", isSafe ? "#38bdf8" : "#4a88f7");
    }
    paintModes();
  }));

  /* ---------- hero line: a 50/50 split between two drafts ---------- */
  const LINES = [
    "Software developer. Film developer. Happiest somewhere with no signal.",
    "I develop things. Software by day, film by night, trails on the weekend.",
  ];
  const lineIdx = params.get("line") === "2" ? 1 : params.get("line") === "1" ? 0 : Math.random() < 0.5 ? 0 : 1;
  $("#line").textContent = LINES[lineIdx];

  /* ---------- develop: name through Redaction's cuts, print through the shader ---------- */
  const nameEl = $("#name");
  const timer = $("#timer");
  const DEV_MS = 2600;
  let devStart = Infinity;
  const develop = () => {
    devStart = performance.now();
    if (reduce) { devStart -= DEV_MS; nameEl.className = "name"; return; }
    ["s100", "s70", "s35", ""].forEach((c, i) =>
      setTimeout(() => (nameEl.className = "name " + c), i * 300));
  };
  nameEl.classList.add("s100");

  // the hand waves once the name has developed, and again whenever you say hi back
  const wave = $("#wave");
  const waveHello = () => {
    if (reduce) return;
    wave.classList.remove("waving");
    void wave.offsetWidth;
    wave.classList.add("waving");
  };
  $("#hi").addEventListener("pointerenter", waveHello);
  wave.addEventListener("click", waveHello);

  const cv = $("#print-canvas");
  const gl = cv.getContext("webgl", { premultipliedAlpha: false, antialias: false });
  const FRAG = `
    precision highp float;
    uniform sampler2D tex; uniform vec2 res; uniform float t, seed; uniform int mode, converted;
    varying vec2 uv;
    float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
    float lum(vec3 c){ return dot(c, vec3(.299, .587, .114)); }
    void main(){
      vec3 c = texture2D(tex, uv).rgb;
      // halation: highlights in a small ring bleed red-orange into their surroundings
      float h = 0.;
      for (int i = 0; i < 16; i++) {
        float a = float(i) * .3927;
        for (int r = 1; r <= 2; r++) {
          vec2 o = vec2(cos(a), sin(a)) * float(r) * 7. / res;
          h += smoothstep(.72, .98, lum(texture2D(tex, uv + o).rgb));
        }
      }
      vec3 halo = (mode == 2 && converted == 1) ? vec3(.15, .4, 1.) : vec3(1., .3, .12);
      c += halo * (h / 32.) * .55;
      float L = lum(c);
      if (mode == 1) c = vec3(pow(L, 1.08)) * vec3(1., .99, .965);
      if (mode == 2) {
        if (converted == 1) c = vec3(L * .08, L * .22, L) * .85;
        else c = vec3(L, L * .2, L * .1) * .8;
      }
      // development: density builds, darkest tones arrive first
      float k = smoothstep(0., 1., clamp((t * 1.45 - L * .62) / .38, 0., 1.));
      vec3 paper = (mode == 2)
        ? (converted == 1 ? vec3(.03, .06, .20) : vec3(.26, .05, .03))
        : (mode == 1 ? vec3(.922, .918, .902) : vec3(.945, .937, .918));
      c = mix(paper, c, k);
      // grain: re-seeded every frame, strongest in the midtones
      float n = hash(floor(uv * res) + seed) - .5;
      c += n * .075 * (1. - abs(L - .5) * 1.3);
      gl_FragColor = vec4(c, 1.);
    }`;
  const VERT = `attribute vec2 p; varying vec2 uv;
    void main(){ uv = vec2(p.x * .5 + .5, .5 - p.y * .5); gl_Position = vec4(p, 0., 1.); }`;
  const U = {};
  let glOk = false;
  if (gl) {
    const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s; };
    const pr = gl.createProgram();
    gl.attachShader(pr, sh(gl.VERTEX_SHADER, VERT));
    gl.attachShader(pr, sh(gl.FRAGMENT_SHADER, FRAG));
    gl.bindAttribLocation(pr, 0, "p");
    gl.linkProgram(pr);
    glOk = gl.getProgramParameter(pr, gl.LINK_STATUS);
    if (glOk) {
      gl.useProgram(pr);
      gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      ["res", "t", "seed", "mode", "converted"].forEach((n) => (U[n] = gl.getUniformLocation(pr, n)));
    }
  }
  const photo = new Image();
  photo.src = "assets/hike.jpg";
  const fitCanvas = () => {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    cv.width = Math.round(cv.clientWidth * dpr);
    cv.height = Math.round(cv.clientHeight * dpr);
    if (glOk) gl.viewport(0, 0, cv.width, cv.height);
  };
  addEventListener("resize", fitCanvas);

  let heroVisible = true;
  new IntersectionObserver(([e]) => (heroVisible = e.isIntersecting)).observe($("#hero"));

  Promise.all([photo.decode(), document.fonts.ready]).then(() => {
    fitCanvas();
    if (glOk) {
      gl.bindTexture(gl.TEXTURE_2D, gl.createTexture());
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, photo);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    }
    develop();
    heroStickers();
    setTimeout(waveHello, reduce ? 0 : DEV_MS + 200);
    let last = 0;
    let drawn2d = false;
    const frame = (now) => {
      const t = Math.min(1, Math.max(0, (now - devStart) / DEV_MS));
      const secs = Math.round(t * 90);
      timer.textContent = t < 1 ? `Dev ${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, "0")}` : "Dev 1:30 · fixed";
      // grain ticks at 24fps, like film through a gate
      if (glOk && heroVisible && now - last > 41) {
        last = now;
        gl.uniform2f(U.res, cv.width, cv.height);
        gl.uniform1f(U.t, t);
        gl.uniform1f(U.seed, reduce ? 1 : Math.floor(now / 41) % 97);
        gl.uniform1i(U.mode, mode);
        gl.uniform1i(U.converted, converted ? 1 : 0);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      } else if (!glOk && !drawn2d) {
        cv.getContext("2d").drawImage(photo, 0, 0, cv.width, cv.height);
        drawn2d = true;
      }
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  });

  /* ---------- table engine: stickers and prints drift, can be thrown, stay where they land ---------- */
  // one shared pointer, normalised to -1..1 around the viewport centre, for sticker parallax
  const pointer = { x: 0, y: 0 };
  if (matchMedia("(hover: hover)").matches) {
    addEventListener("pointermove", (e) => {
      pointer.x = (e.clientX / innerWidth) * 2 - 1;
      pointer.y = (e.clientY / innerHeight) * 2 - 1;
    }, { passive: true });
  }
  // liquid warp: one displacement filter per sticker, its strength driven by speed
  const SVGNS = "http://www.w3.org/2000/svg";
  const filterDefs = $("svg[aria-hidden]");
  let warpId = 0;
  const makeWarp = () => {
    const id = `warp-${warpId++}`;
    const f = document.createElementNS(SVGNS, "filter");
    f.setAttribute("id", id);
    ["x", "y"].forEach((k) => f.setAttribute(k, "-25%"));
    ["width", "height"].forEach((k) => f.setAttribute(k, "150%"));
    f.innerHTML = `<feTurbulence type="fractalNoise" baseFrequency="0.011 0.019" numOctaves="2" seed="${warpId * 7}" result="t"/>` +
      `<feDisplacementMap in="SourceGraphic" in2="t" scale="0" xChannelSelector="R" yChannelSelector="G"/>`;
    filterDefs.appendChild(f);
    return { id, disp: f.lastChild };
  };

  function makeTable(container, layer, { trail = false, warp = false, parallax = false } = {}) {
    const items = [];
    let running = false;
    let visible = true;
    let z = 10;
    let locked = false;
    const TRAIL = 4, LAG = 3;

    const local = (e) => {
      const r = container.getBoundingClientRect();
      return { x: e.clientX - r.left, y: e.clientY - r.top };
    };

    const add = (el, home, opts = {}) => {
      const mk = (cls) => {
        const d = document.createElement("div");
        d.className = "st" + cls;
        d.style.width = opts.width + "px";
        d.appendChild(cls ? el.cloneNode(true) : el);
        layer.appendChild(d);
        return d;
      };
      const ghosts = trail ? Array.from({ length: TRAIL }, () => mk(" ghost")) : [];
      const node = mk("");
      const it = {
        node, ghosts, home, x: 0, y: 0, vx: 0, vy: 0, r: 0, vr: 0, hx: 0, hy: 0, hr: 0, s: 1,
        ph: Math.random() * 6.28, held: false, free: false, moved: false, hist: [], swallow: null,
        depth: 0.4 + Math.random() * 0.8, warp: warp ? makeWarp() : null, wv: 0, warping: false, inner: el,
      };
      items.push(it);
      place(it, true);

      node.addEventListener("pointerdown", (e) => {
        if (e.button !== 0 || it.swallow || locked) return;
        if (e.target.closest("button")) return;     // e.g. waldo, inside a print
        e.preventDefault();
        try { node.setPointerCapture(e.pointerId); } catch (_) {}
        const p = local(e);
        it.held = true; it.free = false;
        it.dragged = false;
        it.downX = e.clientX; it.downY = e.clientY;
        it.ox = p.x - it.x; it.oy = p.y - it.y;
        it.vx = it.vy = 0;
        node.classList.add("held");
        raise(it);
        wake();
      });
      node.addEventListener("pointermove", (e) => {
        if (!it.held) return;
        if (!it.dragged) {
          if (Math.hypot(e.clientX - it.downX, e.clientY - it.downY) > 8) {
            it.dragged = true;
            it.moved = true;
          } else {
            return;
          }
        }
        const p = local(e);
        const nx = p.x - it.ox, ny = p.y - it.oy;
        it.vx = it.vx * 0.4 + (nx - it.x) * 0.6;
        it.vy = it.vy * 0.4 + (ny - it.y) * 0.6;
        it.x = nx; it.y = ny;
      });
      const drop = (e) => {
        if (!it.held) return;
        const wasDragged = it.dragged;
        it.held = false;
        node.classList.remove("held");
        if (wasDragged) {
          it.free = true;
          it.vr = it.vx * 0.35;
        } else {
          it.vx = it.vy = 0;
          if (it.onClick) it.onClick(e);
        }
      };
      node.addEventListener("pointerup", drop);
      node.addEventListener("pointercancel", () => {
        if (!it.held) return;
        it.held = false;
        node.classList.remove("held");
        if (it.dragged) {
          it.free = true;
          it.vr = it.vx * 0.35;
        }
      });
      return it;
    };

    const raise = (it) => { z++; it.ghosts.forEach((g) => (g.style.zIndex = z)); it.node.style.zIndex = ++z; };

    // homes come from layout anchors, so they follow resizes until someone moves the sticker
    function place(it, first) {
      const h = it.home();
      it.w = it.node.offsetWidth; it.h = it.node.offsetHeight;
      if (!it.moved) { it.hx = h.x; it.hy = h.y; it.hr = h.r; }
      if (first || !it.moved) { it.x = it.hx; it.y = it.hy; it.r = it.hr; }
    }
    let resolve = null;   // optional pass that nudges homes apart after layout
    const relayout = () => {
      items.forEach((it) => place(it, false));
      if (resolve) {
        resolve(items);
        items.forEach((it) => { if (!it.moved && !it.free && !it.held) { it.x = it.hx; it.y = it.hy; } });
      }
    };
    const setResolve = (fn) => { resolve = fn; };
    addEventListener("resize", relayout);

    const slap = (stagger = 0.09, delay = 0) => {
      items.forEach((it, i) => {
        it.node.classList.remove("slap");
        void it.node.offsetWidth;
        it.node.style.setProperty("--d", `${delay + i * stagger}s`);
        if (!reduce) it.node.classList.add("slap");
      });
    };

    const slapItem = (it, delay) => {
      it.node.style.setProperty("--d", `${delay}s`);
      if (!reduce) it.node.classList.add("slap");
    };

    const nudge = (it) => {
      raise(it);
      it.moved = true; it.free = true;
      it.vx = (Math.random() - 0.5) * 26;
      it.vy = -12 - Math.random() * 8;
      it.vr = it.vx * 0.4;
      wake();
    };

    const swallow = (to, done) => {
      const t0 = performance.now();
      items.forEach((it, i) => (it.swallow = { t0: t0 + i * 45, fx: it.x, fy: it.y, to }));
      setTimeout(done, 700 + items.length * 45);
      wake();
    };
    const restore = () => {
      items.forEach((it) => { it.swallow = null; it.s = 1; it.moved = false; it.free = false; it.vx = it.vy = 0; place(it, true); });
      relayout();
      slap(0.07);
    };

    let lastT = performance.now();
    const tick = (now) => {
      const dt = Math.min(2, (now - lastT) / 16.67);
      lastT = now;
      const W = container.clientWidth, H = container.clientHeight;
      for (const it of items) {
        if (it.swallow) {
          const k = Math.max(0, Math.min(1, (now - it.swallow.t0) / 520));
          const e = k * k;
          it.x = it.swallow.fx + (it.swallow.to.x - it.swallow.fx) * e;
          it.y = it.swallow.fy + (it.swallow.to.y - it.swallow.fy) * e;
          it.s = 1 - e;
          it.r += 9 * dt * k;
        } else if (it.held) {
          it.r += (it.hr * 0.3 + it.vx * 0.6 - it.r) * 0.15;
          it.vx *= 0.8; it.vy *= 0.8;               // pointermove refreshes these; holding still lets them settle
        } else if (it.free) {
          it.x += it.vx * dt; it.y += it.vy * dt;
          const f = Math.pow(0.93, dt);
          it.vx *= f; it.vy *= f; it.vr *= f;
          it.r += it.vr * dt;
          const mx = it.w * 0.25, my = it.h * 0.25;
          if (it.x < mx) { it.x = mx; it.vx = Math.abs(it.vx) * 0.5; }
          if (it.x > W - mx) { it.x = W - mx; it.vx = -Math.abs(it.vx) * 0.5; }
          if (it.y < my) { it.y = my; it.vy = Math.abs(it.vy) * 0.5; }
          if (it.y > H - my) { it.y = H - my; it.vy = -Math.abs(it.vy) * 0.5; }
          if (Math.hypot(it.vx, it.vy) < 0.25) {
            it.free = false;
            it.hx = it.x; it.hy = it.y; it.hr = it.r; it.vx = it.vy = 0;
          }
        } else if (!reduce) {
          // idle drift around home: slow, small, never in sync with its neighbours
          const t = now / 1000;
          const px = parallax ? -pointer.x * it.depth * 18 : 0;
          const py = parallax ? -pointer.y * it.depth * 12 : 0;
          const tx = it.hx + Math.sin(t * 0.55 + it.ph) * 7 + px;
          const ty = it.hy + Math.cos(t * 0.43 + it.ph * 1.3) * 8 + py;
          const tr = it.hr + Math.sin(t * 0.37 + it.ph) * 1.8;
          it.vx = (it.vx + (tx - it.x) * 0.02 * dt) * 0.85;
          it.vy = (it.vy + (ty - it.y) * 0.02 * dt) * 0.85;
          it.x += it.vx * dt; it.y += it.vy * dt;
          it.r += (tr - it.r) * 0.06 * dt;
        }
        // squash and stretch along the direction of travel
        const speed = Math.hypot(it.vx, it.vy);
        const k = it.swallow || reduce ? 0 : Math.min(0.24, speed * 0.014);
        let stretch = "";
        if (k > 0.01) {
          const a = (Math.atan2(it.vy, it.vx) * 180) / Math.PI;
          stretch = ` rotate(${a.toFixed(1)}deg) scale(${(1 + k).toFixed(3)},${(1 - k * 0.55).toFixed(3)}) rotate(${(-a).toFixed(1)}deg)`;
        }
        const tf = `translate(${(it.x - it.w / 2).toFixed(1)}px,${(it.y - it.h / 2).toFixed(1)}px)${stretch} rotate(${it.r.toFixed(2)}deg) scale(${it.s.toFixed(3)})`;
        it.node.style.transform = tf;
        if (it.warp) {
          it.wv += (Math.min(30, Math.max(0, speed - 1.2) * 2.4) - it.wv) * 0.22;
          if (it.wv > 0.6 && !reduce) {
            if (!it.warping) { it.inner.style.filter = `url(#${it.warp.id})`; it.warping = true; }
            it.warp.disp.setAttribute("scale", it.wv.toFixed(1));
          } else if (it.warping) {
            it.inner.style.filter = ""; it.warping = false;
          }
        }
        if (it.ghosts.length) {
          it.hist.unshift(tf);
          it.hist.length = TRAIL * LAG + 1;
          const o = it.swallow ? 0 : Math.min(0.42, Math.max(0, speed - 1.5) * 0.03);
          it.ghosts.forEach((g, k) => {
            g.style.transform = it.hist[(k + 1) * LAG] || tf;
            g.style.opacity = (o * (1 - k * 0.2)).toFixed(3);
          });
        }
      }
      if (visible || items.some((it) => it.held || it.free || it.swallow)) requestAnimationFrame(tick);
      else running = false;
    };
    const wake = () => { if (!running) { running = true; lastT = performance.now(); requestAnimationFrame(tick); } };
    new IntersectionObserver(([e]) => { visible = e.isIntersecting; if (visible) wake(); }).observe(container);
    const setLocked = (v) => {
      locked = !!v;
      if (locked) {
        items.forEach((it) => {
          if (it.held) {
            it.held = false;
            it.node.classList.remove("held");
          }
        });
      }
    };

    return { items, add, relayout, slap, slapItem, nudge, swallow, restore, wake, setResolve, setLocked };
  }

  /* ---------- hero stickers: one riso print run ---------- */

  const hero = $("#hero");
  const anchors = () => {
    const hr = hero.getBoundingClientRect();
    const rel = (el) => { const r = el.getBoundingClientRect(); return { l: r.left - hr.left, t: r.top - hr.top, w: r.width, h: r.height }; };
    const L = rel($("#line")), P = rel($("#print"));
    const tw = Math.max(300, P.l - L.l);
    const tt = L.t + L.h;
    // Bounded vertical space that is guaranteed to stay within the viewport fold on any aspect ratio
    const maxBottom = Math.min(hr.height, window.innerHeight - hr.top) - 55;
    const th = Math.max(170, maxBottom - (tt + 18));
    return {
      N: rel($("#name-text")),
      L,
      P,
      H: { l: 0, t: 0, w: hr.width, h: hr.height },
      T: { l: L.l, t: tt + 18, w: tw, h: th },
    };
  };

  // Desktop slots: 12 organically undulating, staggered desk positions (never two straight lines)
  const DESK_SLOTS = [
    // Column 1 (left)
    ["T", 0.05, 0.28, -5],
    ["T", 0.08, 0.62, -4],
    // Column 2
    ["T", 0.26, 0.44,  0],
    ["T", 0.23, 0.82, -4],
    // Column 3
    ["T", 0.42, 0.18, -2],
    ["T", 0.44, 0.65,  2],
    // Column 4
    ["T", 0.58, 0.38,  0],
    ["T", 0.62, 0.84, -8],
    // Column 5
    ["T", 0.76, 0.22,  4],
    ["T", 0.72, 0.66,  0],
    // Column 6 (right, towards #print)
    ["T", 0.90, 0.40,  1],
    ["T", 0.95, 0.74,  0],
  ];

  // Mobile slots: 2 along the right edge of the photo, 5 spread out below as you scroll
  const MOB_SLOTS = [
    // 0: Upper-right of photo frame (overlapping photo border)
    ["P", 0.94, 0.22, 0, 0, -4],
    // 1: Lower-right of photo frame (pushed out to the right margin)
    ["P", 1.02, 0.66, 0, 0,  2],
    // 2: Just below photo caption on the right
    ["P", 0.90, 1.00, 0, 60,  4],
    // 3: Just below photo caption on the left
    ["P", 0.18, 1.00, 0, 78, -3],
    // 4: Center, staggered in the middle space
    ["P", 0.52, 1.00, 0, 195,  1],
    // 5: Lower space on the left
    ["P", 0.16, 1.00, 0, 320, -4],
    // 6: Lower space on the right (reserved for worm, interactive)
    ["P", 0.90, 1.00, 0, 325,  3],
  ];

  // [key, width at 1440, baseRotation]
  const STICKERS = [
    ["camera",   168,  -5],
    ["cobra",    176,  -4],
    ["terminal", 138,  -4],
    ["mountain", 186,   0],
    ["dog",      158,  -2],
    ["keycap",   106,  22], // +22 cancels intrinsic tilt in source artwork
    ["canister",  78,   0],
    ["muse",     112, -10],
    ["tent",     128,   0],
    ["duck",     118,   4],
    ["worm",     146,   1],
    ["monk",      92,   0],
  ];
  const narrow = () => innerWidth < 760;

  // relax sticker homes: push them off each other and off the type, so any viewport stays tidy.
  // stickers may still overlap a little (they are stickers), and may sit on the photo.
  function spread(items) {
    const isNarrow = narrow();
    const hr = hero.getBoundingClientRect();
    const rect = (el, shrinkY = 0) => {
      const r = el.getBoundingClientRect();
      return { x: r.left - hr.left + r.width / 2, y: r.top - hr.top + r.height / 2, w: r.width, h: r.height * (1 - shrinkY) };
    };
    const textRect = (el) => { const rg = document.createRange(); rg.selectNodeContents(el); return rg; };
    const zone = (rg, shrinkY = 0) => {
      const r = rg.getBoundingClientRect();
      return { x: r.left - hr.left + r.width / 2, y: r.top - hr.top + r.height / 2, w: r.width + 16, h: r.height * (1 - shrinkY) + 12 };
    };
    const zones = [zone(textRect($("#name-text")), 0.25), zone(textRect($("#hi"))), zone(textRect($("#line"))), rect($(".print figcaption"))];
    const W = hero.clientWidth, H = hero.clientHeight;
    const free = items.filter((it) => !it.moved);
    const push = (a, b, fixedB, k) => {
      const ox = (a.w + b.w) / 2 * k - Math.abs(a.hx - b.x);
      const oy = (a.h + b.h) / 2 * k - Math.abs(a.hy - b.y);
      if (ox <= 0 || oy <= 0) return;
      const m = fixedB ? 1 : 0.5;
      if (ox < oy) { const d = (a.hx < b.x ? -ox : ox) * m; a.hx += d; if (!fixedB) b.x -= d; }
      else { const d = (a.hy < b.y ? -oy : oy) * m; a.hy += d; if (!fixedB) b.y -= d; }
    };
    for (let iter = 0; iter < 60; iter++) {
      for (let i = 0; i < free.length; i++) {
        const a = free[i];
        for (const z of zones) push(a, z, true, 1);
        for (const o of items) {
          if (o === a) continue;
          push(a, { x: o.hx, y: o.hy, w: o.w, h: o.h }, o.moved, isNarrow ? 0.82 : 0.7);
        }
        if (isNarrow) {
          const screenLeft = -hr.left;
          const screenRight = window.innerWidth - hr.left;
          a.hx = Math.max(screenLeft + a.w * 0.5 + 4, Math.min(screenRight - a.w * 0.5 - 4, a.hx));
        } else {
          const marginX = a.w * 0.3;
          a.hx = Math.max(marginX, Math.min(W - marginX, a.hx));
        }
        a.hy = Math.max(a.h * 0.3, Math.min(H - a.h * 0.25, a.hy));
      }
    }
  }

  let heroTable = null;
  function heroStickers() {
    const layer = $("#hero-layer");
    heroTable = makeTable(hero, layer, { trail: true, warp: true, parallax: true });
    const isNarrow = narrow();
    const a0 = anchors();
    const vScale = Math.min(1.0, a0.T.h / 280);
    const scale = isNarrow ? 0.92 : Math.max(0.68, Math.min(1.05, (hero.clientWidth / 1350) * vScale));
    const shuffle = (arr) => {
      const a = [...arr];
      for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
      }
      return a;
    };

    let list = [];
    if (isNarrow) {
      // Mobile: 7 stickers total. Worm is ALWAYS present (in slot 6, lower-right), plus 6 randomly picked stickers
      const wormSticker = STICKERS.find(([k]) => k === "worm");
      const otherStickers = shuffle(STICKERS.filter(([k]) => k !== "worm"));
      const picked = [...otherStickers.slice(0, 6), wormSticker];
      list = picked.map(([key, w, rot], i) => {
        const slot = MOB_SLOTS[i];
        const r = rot + slot[5] * 0.4 + (Math.random() - 0.5) * 4;
        const target = [slot[0], slot[1], slot[2], slot[3], slot[4], r];
        return { key, w, target };
      });
    } else {
      // Desktop: shuffle all 12 stickers and assign them to the 12 organically staggered desk slots
      const shuffledStickers = shuffle(STICKERS);
      list = shuffledStickers.map(([key, w, rot], i) => {
        const slot = DESK_SLOTS[i];
        const r = rot + slot[3] * 0.4 + (Math.random() - 0.5) * 4;
        const target = [slot[0], slot[1], slot[2], 0, 0, r];
        return { key, w, target };
      });
    }

    const imgs = list.map(({ key }) => {
      const img = new Image();
      img.src = `assets/stickers/riso-${key}.webp`;
      img.alt = "";
      img.draggable = false;
      return img;
    });

    // add them all at once, settle the layout, then slap them down one by one
    Promise.all(imgs.map((img) => img.decode().catch(() => {}))).then(() => {
      list.forEach(({ key, w, target }, i) => {
        const wrap = document.createElement("div");
        wrap.className = "in";
        wrap.appendChild(imgs[i]);
        const home = () => {
          const a = anchors();
          const A = a[target[0]];
          return { x: A.l + A.w * target[1] + target[3], y: A.t + A.h * target[2] + target[4], r: target[5] };
        };
        const it = heroTable.add(wrap, home, { width: Math.round(w * scale) });
        it.key = key;
        if (key === "worm") {
          it.onClick = () => {
            if (!wormBusy) summonWorm();
          };
        }
      });
      heroTable.setResolve(spread);
      heroTable.relayout();
      // slap them down in randomized order on every page load
      const slapOrder = shuffle([...heroTable.items]);
      slapOrder.forEach((it, i) => heroTable.slapItem(it, 0.3 + i * 0.07));
    });
    setTimeout(() => heroTable && heroTable.relayout(), 1200);   // late font metrics
  }

  /* thumper: drum four steady beats on the empty table and the worm answers */
  const beats = [];
  let wormBusy = false;
  hero.addEventListener("pointerdown", (e) => {
    if (e.target.closest(".st, a, button")) return;
    const now = performance.now();
    beats.push(now);
    while (beats.length > 4) beats.shift();
    if (beats.length < 4 || wormBusy || !heroTable) return;
    const gaps = beats.slice(1).map((b, i) => b - beats[i]);
    const steady = Math.max(...gaps) / Math.min(...gaps) < 1.4 && gaps.every((g) => g > 220 && g < 950);
    if (steady) { beats.length = 0; summonWorm(); }
  });

  function summonWorm() {
    if (wormBusy || !heroTable) return;
    wormBusy = true;
    heroTable.setLocked(true);
    const worm = document.createElement("div");
    worm.className = "sandworm";
    worm.innerHTML = `<img src="assets/stickers/riso-worm.webp" alt="">`;
    hero.appendChild(worm);
    rumble();
    requestAnimationFrame(() => requestAnimationFrame(() => worm.classList.add("up")));
    if (!reduce) hero.classList.add("quake");
    setTimeout(() => {
      const hr = hero.getBoundingClientRect(), wr = worm.getBoundingClientRect();
      const to = { x: wr.left - hr.left + wr.width * 0.62, y: wr.top - hr.top + wr.height * 0.22 };
      heroTable.swallow(to, () => {
        worm.classList.remove("up");
        hero.classList.remove("quake");
        setTimeout(() => {
          worm.remove();
          heroTable.restore();
          heroTable.setLocked(false);
          wormBusy = false;
        }, 800);
      });
    }, 650);
    console.log("%cThe spice must flow.", "color:#d2321e;font-family:monospace");
  }

  let audioCtx;
  const ctx = () => {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === "suspended") audioCtx.resume();
    return audioCtx;
  };
  function rumble() {
    try {
      const a = ctx(), t = a.currentTime;
      const len = a.sampleRate * 1.6;
      const buf = a.createBuffer(1, len, a.sampleRate);
      const d = buf.getChannelData(0);
      for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.sin((i / len) * Math.PI);
      const src = a.createBufferSource();
      src.buffer = buf;
      const lp = a.createBiquadFilter();
      lp.type = "lowpass"; lp.frequency.value = 120;
      const g = a.createGain();
      g.gain.value = 0.5;
      src.connect(lp); lp.connect(g); g.connect(a.destination);
      src.start(t);
    } catch (e) { /* no audio, the worm comes anyway */ }
  }

  /* ---------- off the clock: tossable prints ---------- */
  const clock = $("#off-the-clock");
  const clockTable = makeTable(clock, $("#clock-layer"));
  const WALDO = `<svg viewBox="0 0 14 20" aria-hidden="true"><circle cx="7" cy="1.6" r="1.4" fill="#f4f1e9"/><rect x="3.2" y="2.4" width="7.6" height="1.9" rx=".6" fill="#e03131"/><rect x="4.1" y="5.6" width="5.8" height="3.9" rx="1.9" fill="#eecfa8"/><rect x="3.2" y="9.7" width="7.6" height="1.4" fill="#e03131"/><rect x="3.2" y="11.1" width="7.6" height="1.4" fill="#f4f1e9"/><rect x="3.2" y="12.5" width="7.6" height="1.4" fill="#e03131"/><rect x="3.8" y="15.3" width="6.4" height="3.4" fill="#2b4c9b"/></svg>`;
  const PRINTS = {
    hike: { src: "assets/hike.jpg", w: 800, h: 1200, width: 236, cap: "PNW, best hiking buddy",
      alt: "A dog leading the way on a trail above an alpine lake", at: [0.3, 0.3, -6] },
    camp: { src: "assets/camp.jpg", w: 1130, h: 1200, width: 270, cap: "Camp, somewhere quiet",
      alt: "Tents pitched beside a deep green alpine lake", at: [0.74, 0.26, 5], waldo: [24.5, 77] },
    paddle: { src: "assets/paddle.jpg", w: 1200, h: 678, width: 340, cap: "Lake Tahoe, glass water",
      alt: "Two paddle boarders on glassy Lake Tahoe below snowy mountains", at: [0.5, 0.76, -3] },
  };
  const tableArea = $("#clock-table");
  const printItems = {};
  Object.entries(PRINTS).forEach(([key, p]) => {
    const fig = document.createElement("figure");
    fig.className = "photo in";
    fig.style.margin = "0";
    fig.innerHTML = `<div style="position:relative"><img src="${p.src}" width="${p.w}" height="${p.h}" alt="${p.alt}" draggable="false"></div><figcaption class="mono">${p.cap}</figcaption>`;
    if (p.waldo) {
      const b = document.createElement("button");
      b.className = "waldo";
      b.setAttribute("aria-label", "Wait, is that Waldo?");
      b.style.left = p.waldo[0] + "%";
      b.style.top = p.waldo[1] + "%";
      b.innerHTML = WALDO + `<span class="waldo__tag mono">you found Waldo!</span>`;
      let waldoTimer;
      b.addEventListener("click", () => {
        b.classList.remove("found");
        void b.offsetWidth;
        b.classList.add("found");
        console.log("%c( you found Waldo! )", "color:#e03131;font-family:monospace");
        clearTimeout(waldoTimer);
        waldoTimer = setTimeout(() => b.classList.remove("found"), 2600);
      });
      fig.firstElementChild.appendChild(b);
    }
    const home = () => {
      const cr = clock.getBoundingClientRect(), tr = tableArea.getBoundingClientRect();
      return { x: tr.left - cr.left + tr.width * p.at[0], y: tr.top - cr.top + tr.height * p.at[1], r: p.at[2] };
    };
    const width = Math.round(p.width * (narrow() ? 0.66 : 1));
    printItems[key] = clockTable.add(fig, home, { width });
  });
  // prints land on the table the first time it scrolls into view
  $$("#clock-layer .st").forEach((n) => (n.style.visibility = "hidden"));
  // decode early so a print never lands as an empty, grainy frame
  const printsReady = Promise.all($$("#clock-layer img").map((i) => i.decode().catch(() => {})));
  new IntersectionObserver(([e], io) => {
    if (!e.isIntersecting) return;
    io.disconnect();
    printsReady.then(() => {
      clockTable.relayout();
      $$("#clock-layer .st").forEach((n) => (n.style.visibility = ""));
      clockTable.slap(0.16, 0.05);
    });
  }, { threshold: 0.2 }).observe(clock);
  $$(".toss").forEach((b) => b.addEventListener("click", () => clockTable.nudge(printItems[b.dataset.print])));

  /* ---------- showcase: contact sheet + print dialog ---------- */
  const v = (src, poster, w, h, cls = "") => ({ type: "video", src, poster, w, h, cls });
  const im = (src, w, h, cls = "", alt = "") => ({ type: "img", src, w, h, cls, alt });
  const WORK = [
    {
      title: "Muse Spark", code: "01A", still: "frame-spark", credit: "Media from ai.meta.com",
      body: "Meta's frontier model. I work on building the supervised fine tuning (SFT) pipelines and benchmarking systems behind it.",
      link: ["Read the announcement", "https://ai.meta.com/blog/introducing-muse-spark-msl/"],
      media: [
        v("muse-v2", "muse-v2-poster", 1080, 1080), v("muse-v3", "muse-v3-poster", 1080, 1080),
        im("muse-1", 1600, 900, "wide", "Scaling observations from RL training"),
        v("muse-v1", "muse-v1-poster", 1080, 1080), v("muse-v0", "muse-v0-poster", 1080, 1080),
        im("muse-2", 1600, 1600, "", "Held-out codebase perplexity chart"), v("muse-v4", "muse-v4-poster", 1080, 1080),
      ],
    },
    {
      title: "Business AI", code: "02A", still: "frame-bell", credit: "Media from about.fb.com",
      body: "I worked on Meta's Business AI agents and outcome measurement, allowing for general availability.",
      link: ["Read the announcement", "https://about.fb.com/news/2026/06/meta-business-agent/"],
      media: [
        im("bizai-0", 1536, 864, "wide", "Meta Business Agent responds to customers 24/7"),
        im("bizai-1", 1600, 1393, "third", "Business Agent in Messenger"),
        im("bizai-2", 1600, 1393, "third", "Business Agent in WhatsApp"),
        im("bizai-3", 1600, 1393, "third", "Business Agent in Instagram"),
        im("bizai-4", 600, 337, "wide", "Business AI expansion in Mexico and on websites"),
      ],
    },
    {
      title: "Marketing Messages", code: "03A", still: "frame-letters", credit: "Media from business.whatsapp.com",
      body: "Metrics and data flows for business messaging on WhatsApp. Modeled reporting under privacy constraints unblocked $6B in incremental revenue.",
      link: ["Developer docs", "https://developers.facebook.com/documentation/business-messaging/whatsapp/marketing-messages/overview"],
      media: [
        im("mm-4", 1000, 653, "wide", "Marketing Messages recurring promotions and product updates in Messenger"),
        im("mm-5", 1024, 464, "wide", "Marketing Messages promotional card deals and catalogs"),
        im("mm-0", 1000, 667, "", "A small business owner preparing orders"),
        im("mm-1", 1000, 667, "", "A customer opening a package at home"),
        im("mm-2", 900, 600, "", "A shop owner in her store"),
        im("mm-3", 900, 600, "", "A business owner working on a laptop"),
      ],
    },
    {
      title: "Commerce Insights", code: "04A", still: "frame-bags", credit: "Media from facebook.com/business",
      body: "One consistent view of commerce across many stores, built on HLL sketches, query tuning and re-modeled views that cut months off development.",
      link: ["Developer docs", "https://developers.facebook.com/documentation/ads-commerce/commerce-platform"],
      media: [
        im("commerce-2", 1024, 565, "wide", "Meta Business Insights performance overview dashboard"),
        im("commerce-3", 1024, 576, "wide", "Meta Business Suite Insights Results graph"),
        im("commerce-0", 1414, 946, "fit", "A clothing rack in a shop"),
        im("commerce-1", 600, 322, "fit", "A food truck business"),
      ],
    },
    {
      title: "Ads Measurement", code: "05A", still: "frame-tape", credit: "Media from facebook.com/business",
      body: "A graph data-model redesign and Spark tuning for measurement pipelines at ads scale. Compute down 95%, storage down 96%.",
      link: ["Developer docs", "https://developers.facebook.com/documentation/ads-commerce/conversions-api"],
      media: [
        im("capi-0", 1424, 1138, "", "A conversion chart over a product photo"),
        v("capi-v0", null, 713, 572),
      ],
    },
    { title: "Arrakis Runner", code: "06A", still: "frame-game", game: true },
  ];

  const LOOP = '<svg class="loop" viewBox="0 0 300 220" preserveAspectRatio="none" aria-hidden="true"><path pathLength="1" d="M60 20 C138 4 264 7 285 66 C303 128 276 205 153 209 C39 213 6 158 15 97 C21 46 81 13 180 18"/></svg>';
  const frames = $("#frames");
  const dlg = $("#print-dialog");
  const still = $("#pd-still");
  const mediaBox = $("#pd-media");
  let openImg = null;
  const transition = (fn) => (document.startViewTransition && !reduce ? document.startViewTransition(fn) : (fn(), null));

  const renderMedia = (w) => {
    mediaBox.innerHTML = "";
    w.media.forEach((m) => {
      let el;
      if (m.type === "video") {
        el = document.createElement("video");
        el.muted = true; el.loop = true; el.playsInline = true; el.preload = "none";
        el.setAttribute("muted", "");
        el.dataset.src = `assets/work/${m.src}.mp4`;
        if (m.poster) el.poster = `assets/work/${m.poster}.webp`;
        el.setAttribute("aria-hidden", "true");
      } else {
        el = document.createElement("img");
        el.src = `assets/work/${m.src}.webp`;
        el.alt = m.alt;
        el.loading = "lazy";
        el.decoding = "async";
      }
      el.width = m.w; el.height = m.h;
      if (m.cls === "wide") el.className = "wide";
      if (m.cls === "third") el.style.gridColumn = "span 1";
      if (m.cls === "fit") { el.style.aspectRatio = "3 / 2"; el.style.objectFit = "cover"; }
      mediaBox.appendChild(el);
    });
    mediaBox.style.gridTemplateColumns = w.media.some((m) => m.cls === "third") ? "repeat(3, minmax(0, 1fr))" : "";
    $$(".wide", mediaBox).forEach((n) => (n.style.gridColumn = "1 / -1"));
  };

  const openPrint = (w, img) => {
    openImg = img;
    img.style.viewTransitionName = "print";
    transition(() => {
      img.style.viewTransitionName = "";
      still.src = img.src;
      still.alt = "";
      $("#pd-edge").textContent = `Frame ${w.code}`;
      $("#pd-title").textContent = w.title;
      $("#pd-body").textContent = w.body;
      $("#pd-link").innerHTML = `<a href="${w.link[1]}" target="_blank" rel="noopener noreferrer">${w.link[0]}</a>`;
      $("#pd-credit").textContent = w.credit;
      renderMedia(w);
      still.style.viewTransitionName = "print";
      dlg.showModal();
      $(".pd", dlg).scrollTop = 0;
      $$("video", mediaBox).forEach((vid) => { vid.src = vid.dataset.src; vid.play().catch(() => {}); });
    });
  };
  const closePrint = () => {
    $$("video", mediaBox).forEach((vid) => { vid.pause(); vid.removeAttribute("src"); vid.load(); });
    const t = transition(() => {
      still.style.viewTransitionName = "";
      dlg.close();
      if (openImg) openImg.style.viewTransitionName = "print";
    });
    const done = () => { if (openImg) { openImg.style.viewTransitionName = ""; openImg.closest(".fr").focus({ preventScroll: true }); } };
    t ? t.finished.then(done, done) : done();
  };
  $("#pd-close").addEventListener("click", closePrint);
  dlg.addEventListener("cancel", (e) => { e.preventDefault(); closePrint(); });
  dlg.addEventListener("click", (e) => { if (e.target === dlg) closePrint(); });

  WORK.forEach((w) => {
    const b = document.createElement("button");
    b.className = "fr";
    b.setAttribute("aria-label", w.game ? "Play Arrakis Runner" : `Open ${w.title}`);
    b.innerHTML = `<div class="pic"><img src="assets/work/${w.still}.webp" alt="" ${w.game ? 'class="px"' : ""} loading="lazy" width="960" height="640"></div>` +
      `<div class="edge mono"><span>${w.title}</span><span>▸ ${w.code}</span></div>${LOOP}`;
    b.addEventListener("click", () => {
      if (w.game) { if (window.__worm) window.__worm.open(b); return; }
      openPrint(w, $("img", b));
    });
    frames.appendChild(b);
  });

  /* ---------- say hi: click to copy ---------- */
  const email = $("#email");
  const emailState = $(".email__state", email);
  let copyTimer;
  email.addEventListener("click", async () => {
    const addr = email.dataset.email;
    let ok = false;
    try { await navigator.clipboard.writeText(addr); ok = true; } catch (e) {
      const ta = document.createElement("textarea");
      ta.value = addr; ta.setAttribute("readonly", ""); ta.style.position = "fixed"; ta.style.opacity = "0";
      document.body.appendChild(ta); ta.select();
      try { ok = document.execCommand("copy"); } catch (err) { ok = false; }
      ta.remove();
    }
    if (!ok) {
      const range = document.createRange();
      range.selectNodeContents($(".email__addr", email));
      getSelection().removeAllRanges(); getSelection().addRange(range);
    }
    email.classList.toggle("copied", ok);
    emailState.textContent = ok ? "Copied to clipboard" : "Selected, press ⌘C to copy";
    clearTimeout(copyTimer);
    copyTimer = setTimeout(() => { email.classList.remove("copied"); emailState.textContent = "Click to copy"; }, 2400);
  });

  /* ---------- easter eggs ---------- */
  // "Wololo." doubles as a typing hint: the next letter blinks, typed letters turn red
  const woloWord = $(".wolo-word");
  const woloLtrs = [];
  if (woloWord) {
    const txt = woloWord.textContent;
    woloWord.setAttribute("aria-label", txt);
    woloWord.textContent = "";
    [...txt].forEach((ch) => {
      const s = document.createElement("span");
      s.textContent = ch;
      s.setAttribute("aria-hidden", "true");
      if (/[a-z]/i.test(ch)) { s.className = "wolo-ltr"; woloLtrs.push(s); }
      woloWord.appendChild(s);
    });
  }
  const setWolo = (n, done) => woloLtrs.forEach((s, i) => {
    s.classList.toggle("lit", done || i < n);
    s.classList.toggle("next", !done && i === n);
  });
  setWolo(0, false);

  let woloAudio;
  const playWololo = () => {
    try {
      woloAudio = woloAudio || Object.assign(new Audio("assets/Wololo.mp3"), { volume: 0.55 });
      woloAudio.currentTime = 0;
      woloAudio.play().catch(() => {});
    } catch (e) { /* silent conversion */ }
  };

  // wololo: the monk walks on, casts, every grease-pencil mark changes team, then he walks off
  let monkBusy = false;
  const convert = () => {
    converted = !converted;
    root.classList.toggle("converted", converted);
    if (converted) {
      const isSafe = root.classList.contains("safe");
      root.style.setProperty("--red", isSafe ? "#60a5fa" : "#2f62d6");
      root.style.setProperty("--amber", isSafe ? "#38bdf8" : "#4a88f7");
    } else {
      root.style.removeProperty("--red");
      root.style.removeProperty("--amber");
    }
    updateThemeColor();
  };
  const wololo = () => {
    if (reduce || monkBusy) { convert(); playWololo(); return; }
    monkBusy = true;
    const m = document.createElement("div");
    m.className = "monk-run walking";
    m.setAttribute("aria-hidden", "true");
    m.innerHTML = `<img src="assets/stickers/riso-monk.webp" alt=""><span class="bubble mono">Wololo!</span><span class="ring"></span><span class="ring"></span><span class="ring"></span>`;
    document.body.appendChild(m);
    const stop = innerWidth * 0.42;
    const walk = m.animate([
      { transform: "translateX(-120px)" },
      { transform: `translateX(${stop}px)`, offset: 0.36 },
      { transform: `translateX(${stop}px)`, offset: 0.64 },
      { transform: `translateX(${innerWidth + 120}px)` },
    ], { duration: 6400, fill: "forwards" });
    setTimeout(() => { m.classList.replace("walking", "casting"); playWololo(); setTimeout(convert, 250); }, 2300);
    setTimeout(() => m.classList.replace("casting", "walking"), 4100);
    walk.finished.then(() => { m.remove(); monkBusy = false; });
  };

  // how do you turn this on: the Cobra tears across the bottom of the page
  let carBusy = false;
  const vroom = () => {
    try {
      const a = ctx(), t = a.currentTime;
      const o = a.createOscillator(), lp = a.createBiquadFilter(), g = a.createGain();
      o.type = "sawtooth";
      o.frequency.setValueAtTime(55, t);
      o.frequency.exponentialRampToValueAtTime(190, t + 0.9);
      o.frequency.exponentialRampToValueAtTime(120, t + 1.1);
      o.frequency.exponentialRampToValueAtTime(260, t + 2);
      lp.type = "lowpass"; lp.frequency.value = 700;
      g.gain.setValueAtTime(0.0001, t);
      g.gain.exponentialRampToValueAtTime(0.09, t + 0.15);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 2.4);
      o.connect(lp); lp.connect(g); g.connect(a.destination);
      o.start(t); o.stop(t + 2.5);
    } catch (e) { /* silent drive-by */ }
  };
  const cobra = () => {
    if (reduce || carBusy) return;
    carBusy = true;
    const c = document.createElement("div");
    c.className = "cobra-run";
    c.setAttribute("aria-hidden", "true");
    c.innerHTML = `<div class="cobra-run__lines"></div><img src="assets/stickers/riso-cobra.webp" alt="">`;
    document.body.appendChild(c);
    const drive = c.animate([{ transform: "translateX(-300px)" }, { transform: `translateX(${innerWidth + 320}px)` }],
      { duration: 2900, easing: "cubic-bezier(.55, 0, .85, .45)", fill: "forwards" });
    const puffs = setInterval(() => {
      const r = c.getBoundingClientRect();
      const p = document.createElement("span");
      p.className = "puff";
      p.style.left = r.left - 4 + "px";
      p.style.top = r.bottom - 34 + "px";
      document.body.appendChild(p);
      p.animate([{ opacity: 0.7, transform: "scale(.4)" }, { opacity: 0, transform: "translate(-30px,-14px) scale(2.2)" }], { duration: 750, easing: "ease-out" })
        .finished.then(() => p.remove());
    }, 80);
    vroom();
    drive.finished.then(() => { clearInterval(puffs); c.remove(); carBusy = false; });
  };

  let typed = "";
  addEventListener("keydown", (e) => {
    if (e.key.length !== 1 || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.target.closest && e.target.closest("input, textarea")) return;
    typed = (typed + e.key.toLowerCase()).slice(-30);
    let prog = 0;
    for (let n = 6; n > 0; n--) if (typed.endsWith("wololo".slice(0, n))) { prog = n; break; }
    setWolo(prog === 6 ? 6 : prog, prog === 6);
    if (typed.endsWith("wololo")) {
      typed = "";
      setTimeout(() => setWolo(0, false), 1500);
      wololo();
    } else if (typed.endsWith("how do you turn this on")) {
      typed = "";
      cobra();
    }
  });

  console.log(
    "%cVasan%c\n\nYou found the console. Two more:\n· type  w o l o l o  anywhere\n· drum four steady beats on the light table\n\nsreenivasan.ramesh@gmail.com",
    "font: 40px Georgia, serif; color: #151311;",
    "font: 12px monospace; color: #6f6a62;"
  );
})();
