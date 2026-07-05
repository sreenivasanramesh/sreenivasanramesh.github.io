/* Vasan Ramesh — interactions
   Everything degrades: reduced-motion users get a static, fully readable page. */

(() => {
  "use strict";

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const finePointer = window.matchMedia("(hover: hover) and (pointer: fine)").matches;

  /* ---------- split [data-letters] into animatable spans ---------- */
  if (!reduceMotion) {
    document.querySelectorAll("[data-letters]").forEach((el) => {
      const text = el.textContent;
      el.textContent = "";
      el.setAttribute("aria-label", text.trim());
      [...text].forEach((ch, i) => {
        const s = document.createElement("span");
        s.className = "ltr";
        s.style.setProperty("--i", i);
        s.textContent = ch === " " ? " " : ch;
        s.setAttribute("aria-hidden", "true");
        el.appendChild(s);
      });
    });
  }

  /* ---------- letter-hop spans for the contact CTA ---------- */
  /* letters grouped per word so lines can only break at spaces */
  document.querySelectorAll("[data-letters-hover]").forEach((el) => {
    const text = el.textContent;
    el.textContent = "";
    el.setAttribute("aria-label", text.trim());
    let i = 0;
    text.split(" ").forEach((word, w, words) => {
      const wordEl = document.createElement("span");
      wordEl.className = "hop-word";
      wordEl.setAttribute("aria-hidden", "true");
      [...word].forEach((ch) => {
        const s = document.createElement("span");
        s.className = "hop";
        s.style.setProperty("--i", i++);
        s.textContent = ch;
        wordEl.appendChild(s);
      });
      el.appendChild(wordEl);
      if (w < words.length - 1) el.appendChild(document.createTextNode(" "));
    });
  });

  /* ---------- scroll reveals ---------- */
  const revealEls = document.querySelectorAll("[data-reveal]");
  // stagger siblings that share a parent
  const groups = new Map();
  revealEls.forEach((el) => {
    const parent = el.parentElement;
    if (!groups.has(parent)) groups.set(parent, 0);
    const idx = groups.get(parent);
    el.style.setProperty("--d", `${Math.min(idx * 0.09, 0.45)}s`);
    groups.set(parent, idx + 1);
  });

  if ("IntersectionObserver" in window && !reduceMotion) {
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.add("is-in");
            io.unobserve(e.target);
          }
        });
      },
      { threshold: 0.12, rootMargin: "0px 0px -8% 0px" }
    );
    revealEls.forEach((el) => io.observe(el));
  } else {
    revealEls.forEach((el) => el.classList.add("is-in"));
  }

  /* ---------- experience accordion ---------- */
  document.querySelectorAll(".job__row").forEach((btn) => {
    btn.addEventListener("click", () => {
      const job = btn.closest(".job");
      const open = job.classList.toggle("open");
      btn.setAttribute("aria-expanded", String(open));
    });
  });

  /* ---------- marquee: fill track, then double for seamless -50% loop ---------- */
  const setupMarquees = () => {
    document.querySelectorAll(".marquee").forEach((marquee) => {
      const track = marquee.querySelector(".marquee__track");
      if (!track.dataset.unit) track.dataset.unit = track.innerHTML;
      track.innerHTML = track.dataset.unit;
      let width = track.scrollWidth;
      const target = marquee.clientWidth * 1.25;
      let guard = 0;
      while (width < target && guard < 10) {
        track.innerHTML += track.dataset.unit;
        width = track.scrollWidth;
        guard++;
      }
      track.innerHTML += track.innerHTML; // exact duplicate → translateX(-50%) loops clean
      // constant speed regardless of track length or viewport, so both rows
      // (and mobile) glide at the same rate
      const PX_PER_SEC = marquee.classList.contains("marquee--alt") ? 52 : 62;
      track.style.animationDuration = track.scrollWidth / 2 / PX_PER_SEC + "s";
    });
  };
  setupMarquees();
  // widths change when the web fonts swap in; measure again once they're ready
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(setupMarquees);
  }

  /* ---------- clocks (Seattle) ---------- */
  const clockEls = document.querySelectorAll("[data-clock]");
  const fmt = new Intl.DateTimeFormat("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "America/Los_Angeles",
  });
  const tick = () => clockEls.forEach((el) => (el.textContent = fmt.format(new Date())));
  tick();
  setInterval(tick, 10_000);

  /* ---------- scroll progress ---------- */
  const progress = document.querySelector(".progress");
  let progressQueued = false;
  const paintProgress = () => {
    const doc = document.documentElement;
    const max = doc.scrollHeight - window.innerHeight;
    progress.style.transform = `scaleX(${max > 0 ? window.scrollY / max : 0})`;
    progressQueued = false;
  };
  window.addEventListener(
    "scroll",
    () => {
      if (!progressQueued) {
        progressQueued = true;
        requestAnimationFrame(paintProgress);
      }
    },
    { passive: true }
  );
  paintProgress();

  /* ---------- off-the-clock: tossable photo cards ---------- */
  const PHOTOS = {
    hike: {
      src: "assets/hike.jpg", thumb: "assets/hike-thumb.jpg", w: 800, h: 1200,
      cardW: 220, alt: "Hiking trail above an alpine lake in the PNW, a dog leading the way",
      cap: "PNW trails, best hiking buddy",
    },
    camp: {
      src: "assets/camp.jpg", thumb: "assets/camp-thumb.jpg", w: 1130, h: 1200,
      cardW: 250, alt: "Tent pitched beside a deep green alpine lake",
      cap: "camp, somewhere quiet",
    },
    paddle: {
      src: "assets/paddle.jpg", thumb: "assets/paddle-thumb.jpg", w: 1200, h: 678,
      cardW: 320, alt: "Two paddle boarders on glassy Lake Tahoe below snowy mountains",
      cap: "Lake Tahoe, glass water",
    },
    aoe: { text: true, cardW: 210, cap: "one more game, I promise" },
  };

  const layer = document.querySelector(".photo-layer");
  const peek = document.querySelector(".toss-peek");
  const live = new Map(); // key -> card state
  let zTop = 1;
  let physicsRunning = false;

  const buildCard = (key) => {
    const p = PHOTOS[key];
    const fig = document.createElement("figure");
    fig.className = "photo-card" + (p.text ? " photo-card--text" : "");
    const scale = window.innerWidth < 700 ? 0.7 : 1;
    fig.style.setProperty("--w", Math.round(p.cardW * scale) + "px");
    if (p.text) {
      fig.insertAdjacentHTML(
        "afterbegin",
        `<div class="photo-card__shout"><span class="serif">Wololo!</span><small>age of empires ii</small></div>`
      );
    } else {
      const img = document.createElement("img");
      img.src = p.src;
      img.width = p.w;
      img.height = p.h;
      img.alt = p.alt;
      img.draggable = false;
      fig.appendChild(img);
    }
    fig.insertAdjacentHTML(
      "beforeend",
      `<figcaption>${p.cap}</figcaption><button class="photo-card__close" aria-label="Close photo">×</button>`
    );
    return fig;
  };

  const removeCard = (key) => {
    const c = live.get(key);
    if (!c) return;
    live.delete(key);
    c.el.classList.add("leaving");
    setTimeout(() => c.el.remove(), 320);
  };

  let stillFrames = 0;
  const physics = () => {
    if (!live.size) { physicsRunning = false; return; }
    const pad = 14;
    let allSettled = true;
    live.forEach((c) => {
      const w = c.el.offsetWidth, h = c.el.offsetHeight;
      if (c.dragging) {
        c.vx = (c.tx - c.x) * 0.35;
        c.vy = (c.ty - c.y) * 0.35;
        c.x += c.vx;
        c.y += c.vy;
      } else {
        // rubberband: spring back when past the edges
        const minX = pad, minY = pad;
        const maxX = window.innerWidth - w - pad;
        const maxY = window.innerHeight - h - pad;
        if (c.x < minX) c.vx += (minX - c.x) * 0.12;
        if (c.x > maxX) c.vx += (maxX - c.x) * 0.12;
        if (c.y < minY) c.vy += (minY - c.y) * 0.12;
        if (c.y > maxY) c.vy += (maxY - c.y) * 0.12;
        c.vx *= 0.92;
        c.vy *= 0.92;
        c.x += c.vx;
        c.y += c.vy;
      }
      // tilt with horizontal velocity, settle back to base rotation
      const target = c.baseRot + Math.max(-14, Math.min(14, c.vx * 0.7));
      c.rot += (target - c.rot) * 0.12;
      // jelly: squash-and-stretch aligned with the motion direction
      const speed = Math.hypot(c.vx, c.vy);
      const jellyTarget = Math.min(0.2, speed * 0.0075);
      c.jelly = (c.jelly || 0) + (jellyTarget - (c.jelly || 0)) * 0.22;
      if (speed > 1.2) c.jellyAng = Math.atan2(c.vy, c.vx);
      let deform = "";
      if (c.jelly > 0.004 && c.jellyAng !== undefined) {
        deform =
          ` rotate(${c.jellyAng}rad)` +
          ` scale(${1 + c.jelly}, ${1 - c.jelly * 0.65})` +
          ` rotate(${-c.jellyAng}rad)`;
      }
      c.el.style.transform = `translate(${c.x}px, ${c.y}px)${deform} rotate(${c.rot}deg)`;
      if (c.dragging || Math.abs(c.vx) + Math.abs(c.vy) > 0.05 ||
          Math.abs(target - c.rot) > 0.05 || c.jelly > 0.004) {
        allSettled = false;
      }
    });
    // sleep the loop once everything has come to rest
    stillFrames = allSettled ? stillFrames + 1 : 0;
    if (stillFrames > 30) { physicsRunning = false; return; }
    requestAnimationFrame(physics);
  };
  const wakePhysics = () => {
    stillFrames = 0;
    if (!physicsRunning) { physicsRunning = true; requestAnimationFrame(physics); }
  };
  window.addEventListener("resize", () => { if (live.size) wakePhysics(); });

  const nudge = (c) => {
    c.vx += (Math.random() - 0.5) * 30;
    c.vy -= 22;
    c.baseRot = (Math.random() - 0.5) * 10;
    c.el.style.zIndex = ++zTop;
    wakePhysics();
  };

  const toss = (key, fromX, fromY, bias) => {
    if (live.get(key)) {
      // already out: give it a playful nudge instead of duplicating
      nudge(live.get(key));
      return;
    }
    const el = buildCard(key);
    layer.appendChild(el);
    const w = el.offsetWidth, h = el.offsetHeight;
    const cx = window.innerWidth / 2 - w / 2;
    const cy = window.innerHeight / 2 - h / 2;
    const c = {
      el,
      x: reduceMotion ? cx : fromX - w / 2,
      y: reduceMotion ? cy : fromY - h / 2,
      vx: 0, vy: 0, rot: 0,
      baseRot: (Math.random() - 0.5) * 10,
      dragging: false, tx: 0, ty: 0,
    };
    if (!reduceMotion) {
      // toss toward the bias point (or a spot near the viewport center)
      const destX = bias
        ? window.innerWidth * bias.x - w / 2 + (Math.random() - 0.5) * 60
        : cx + (Math.random() - 0.5) * window.innerWidth * 0.3;
      const destY = bias
        ? window.innerHeight * bias.y - h / 2 + (Math.random() - 0.5) * 40
        : cy + (Math.random() - 0.5) * window.innerHeight * 0.2;
      c.vx = (destX - c.x) * 0.09;
      c.vy = (destY - c.y) * 0.09;
      c.rot = (Math.random() - 0.5) * 24;
    } else {
      c.rot = c.baseRot;
      if (bias) {
        c.x = window.innerWidth * bias.x - w / 2;
        c.y = window.innerHeight * bias.y - h / 2;
      }
    }
    el.style.zIndex = ++zTop;
    el.style.transform = `translate(${c.x}px, ${c.y}px) rotate(${c.rot}deg)`;
    live.set(key, c);
    wakePhysics();

    // dragging
    el.addEventListener("pointerdown", (e) => {
      if (e.target.closest(".photo-card__close")) return;
      c.dragging = true;
      c.offX = e.clientX - c.x;
      c.offY = e.clientY - c.y;
      c.tx = c.x; c.ty = c.y;
      c.downX = e.clientX;
      c.downY = e.clientY;
      el.classList.add("grabbing");
      el.style.zIndex = ++zTop;
      el.setPointerCapture(e.pointerId);
      wakePhysics();
    });
    el.addEventListener("pointermove", (e) => {
      if (!c.dragging) return;
      c.tx = e.clientX - c.offX;
      c.ty = e.clientY - c.offY;
    });
    const drop = (e) => {
      if (!c.dragging) return;
      c.dragging = false;
      el.classList.remove("grabbing");
      // a press with no real movement is a tap: bounce instead of drop
      if (e && Math.hypot(e.clientX - c.downX, e.clientY - c.downY) < 6 && !reduceMotion) {
        nudge(c);
      }
    };
    el.addEventListener("pointerup", drop);
    el.addEventListener("pointercancel", () => drop());
    el.querySelector(".photo-card__close").addEventListener("click", () => removeCard(key));
  };

  document.querySelectorAll(".toss").forEach((btn) => {
    const key = btn.dataset.photo;
    btn.addEventListener("click", (e) => {
      const r = btn.getBoundingClientRect();
      toss(key, e.clientX || r.left + r.width / 2, e.clientY || r.top);
      peek.classList.remove("on");
    });
    // hover: preload + thumbnail peek
    btn.addEventListener("mouseenter", () => {
      const p = PHOTOS[key];
      if (!p.thumb) return;
      new Image().src = p.src; // warm the full-size image
      peek.style.backgroundImage = `url(${p.thumb})`;
      peek.classList.add("on");
    });
    btn.addEventListener("mouseleave", () => peek.classList.remove("on"));
  });

  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") [...live.keys()].forEach(removeCard);
  });

  // photos wander in on their own the first time About scrolls into view.
  // no persistence, on purpose: close them all and they return on refresh.
  const AUTO_SPAWN = [
    { key: "hike", bias: { x: 0.68, y: 0.62 } },
    { key: "camp", bias: { x: 0.84, y: 0.4 } },
    { key: "paddle", bias: { x: 0.74, y: 0.82 } },
  ];
  const aboutSection = document.querySelector("#about");
  if (aboutSection && "IntersectionObserver" in window) {
    const spawnIO = new IntersectionObserver(
      (entries) => {
        if (!entries[0].isIntersecting) return;
        spawnIO.disconnect();
        AUTO_SPAWN.forEach(({ key, bias }, i) => {
          setTimeout(() => {
            const btn = document.querySelector(`.toss[data-photo="${key}"]`);
            const r = btn.getBoundingClientRect();
            toss(key, r.left + r.width / 2, r.top + r.height / 2, bias);
          }, 200 + i * 280);
        });
      },
      { threshold: 0.4 }
    );
    spawnIO.observe(aboutSection);
  }

  /* ---------- easter eggs ---------- */

  // 1) AoE2 cheat codes, typed anywhere on the page.
  //    "wololo": conversion wash, the accent flips between team colors,
  //              and a monk glides across the bottom of the screen.
  //    "how do you turn this on": the Shelby Cobra drives by.
  const ACCENTS = ["#ff4d24", "#3a7bff"];
  let accentIdx = 0;
  let typed = "";
  const wash = document.createElement("div");
  wash.className = "wololo-wash";
  const toast = document.createElement("div");
  toast.className = "wololo-toast";
  toast.textContent = "wololo";
  document.body.append(wash, toast);

  const MONK_SVG =
    `<svg width="52" height="64" viewBox="0 0 26 32" fill="currentColor" aria-hidden="true">` +
    `<path d="M13 1c-4 0-7 3-7 7 0 2 .6 3.4 1.4 4.4L4 29c-.3 1.2.5 2 1.6 2h14.8c1.1 0 1.9-.8 1.6-2l-3.4-16.6C19.4 11.4 20 10 20 8c0-4-3-7-7-7z"/>` +
    `<circle cx="13" cy="8" r="3.4" fill="#0e0e0c"/>` +
    `<rect x="22.4" y="6" width="2" height="18" rx="1"/>` +
    `<rect x="20.4" y="9" width="6" height="2" rx="1"/>` +
    `</svg>`;
  const CAR_SVG =
    `<svg width="104" height="42" viewBox="0 0 52 21" shape-rendering="crispEdges" aria-hidden="true">` +
    `<rect x="3" y="9" width="46" height="6" fill="currentColor"/>` +
    `<rect x="0" y="11" width="4" height="4" fill="currentColor"/>` +
    `<rect x="15" y="4" width="16" height="5" fill="currentColor"/>` +
    `<rect x="17" y="5" width="9" height="4" fill="#9fd9ff"/>` +
    `<rect x="46" y="7" width="4" height="3" fill="currentColor"/>` +
    `<circle cx="13" cy="16" r="4.2" fill="#14130f"/><circle cx="13" cy="16" r="1.6" fill="#eae7df"/>` +
    `<circle cx="39" cy="16" r="4.2" fill="#14130f"/><circle cx="39" cy="16" r="1.6" fill="#eae7df"/>` +
    `</svg>`;

  // the "Wololo." word in About doubles as a typing-progress hint:
  // the next expected letter blinks, typed letters lock to the accent color
  const woloWord = document.querySelector(".wolo-word");
  const woloLtrs = [];
  if (woloWord) {
    const txt = woloWord.textContent;
    woloWord.setAttribute("aria-label", txt);
    woloWord.textContent = "";
    [...txt].forEach((ch) => {
      const s = document.createElement("span");
      s.textContent = ch;
      s.setAttribute("aria-hidden", "true");
      if (/[a-z]/i.test(ch)) {
        s.className = "wolo-ltr";
        woloLtrs.push(s);
      }
      woloWord.appendChild(s);
    });
  }
  const setWoloProgress = (n, done) => {
    woloLtrs.forEach((s, i) => {
      s.classList.toggle("lit", done || i < n);
      s.classList.toggle("next", !done && i === n);
    });
  };
  setWoloProgress(0, false);

  // wololo sound: plays assets/Wololo.mp3, falls back to a synthesized chant
  // if the file is missing or playback is blocked. triggered by typing, which
  // counts as user activation, so autoplay policy allows it.
  let woloAudio;
  const playWololo = () => {
    try {
      if (!woloAudio) {
        woloAudio = new Audio("assets/Wololo.mp3");
        woloAudio.preload = "auto";
        woloAudio.volume = 0.55;
      }
      woloAudio.currentTime = 0;
      woloAudio.play().catch(playWololoSynth);
    } catch (e) {
      playWololoSynth();
    }
  };

  let audioCtx;
  const playWololoSynth = () => {
    try {
      audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
      if (audioCtx.state === "suspended") audioCtx.resume();
      const t0 = audioCtx.currentTime + 0.02;
      const master = audioCtx.createGain();
      master.gain.value = 0.13;
      master.connect(audioCtx.destination);
      const syllable = (t, f0, formFrom, formTo, dur) => {
        const osc = audioCtx.createOscillator();
        osc.type = "sawtooth";
        osc.frequency.setValueAtTime(f0, t);
        osc.frequency.linearRampToValueAtTime(f0 * 0.97, t + dur);
        const formant = audioCtx.createBiquadFilter();
        formant.type = "bandpass";
        formant.Q.value = 4;
        formant.frequency.setValueAtTime(formFrom, t);
        formant.frequency.exponentialRampToValueAtTime(formTo, t + dur * 0.55);
        const lp = audioCtx.createBiquadFilter();
        lp.type = "lowpass";
        lp.frequency.value = 1300;
        const g = audioCtx.createGain();
        g.gain.setValueAtTime(0.0001, t);
        g.gain.exponentialRampToValueAtTime(1, t + 0.05);
        g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
        osc.connect(formant); formant.connect(lp); lp.connect(g); g.connect(master);
        osc.start(t);
        osc.stop(t + dur + 0.05);
      };
      syllable(t0, 165, 320, 780, 0.34);         // wo
      syllable(t0 + 0.3, 220, 520, 900, 0.3);    // LO
      syllable(t0 + 0.58, 185, 500, 820, 0.46);  // lo
    } catch (e) { /* no audio, no problem */ }
  };

  const runCheat = (variant, label, svg, lifeMs) => {
    const d = document.createElement("div");
    d.className = `cheat-runner cheat-runner--${variant}`;
    d.innerHTML = `<span class="cheat-runner__label">${label}</span>${svg}`;
    document.body.appendChild(d);
    setTimeout(() => d.remove(), lifeMs);
  };

  window.addEventListener("keydown", (e) => {
    if (e.key.length !== 1) return;
    typed = (typed + e.key.toLowerCase()).slice(-30);
    // hint progress: longest suffix of the buffer that is a prefix of "wololo"
    let woloProg = 0;
    for (let n = 6; n > 0; n--) {
      if (typed.endsWith("wololo".slice(0, n))) { woloProg = n; break; }
    }
    if (woloProg === 6) {
      setWoloProgress(6, true);
      setTimeout(() => setWoloProgress(0, false), 1500);
    } else {
      setWoloProgress(woloProg, false);
    }
    if (typed.endsWith("wololo")) {
      typed = "";
      accentIdx = 1 - accentIdx;
      document.documentElement.style.setProperty("--accent", ACCENTS[accentIdx]);
      document.dispatchEvent(new CustomEvent("vr:accent"));
      document.dispatchEvent(new CustomEvent("vr:shock"));
      if (!reduceMotion) {
        for (const el of [wash, toast]) {
          el.classList.remove("go");
          void el.offsetWidth; // restart animation
          el.classList.add("go");
        }
        runCheat("monk", "wololo", MONK_SVG, 7000);
      }
      playWololo();
    } else if (typed.endsWith("how do you turn this on")) {
      typed = "";
      if (!reduceMotion) runCheat("car", "vroom", CAR_SVG, 4300);
    }
  });

  // hover a work card: a quiet shower of its icon, clipped inside the card
  document.querySelectorAll(".card[data-icon]").forEach((card) => {
    const iconSvg = card.querySelector(".card__icon svg");
    if (!iconSvg) return;
    let lastRain = 0;
    card.addEventListener("mouseenter", () => {
      if (reduceMotion) return;
      const now = Date.now();
      if (now - lastRain < 1200) return;
      lastRain = now;
      for (let i = 0; i < 7; i++) {
        const s = document.createElement("span");
        s.className = "glyph-drop";
        s.appendChild(iconSvg.cloneNode(true));
        s.style.left = 5 + Math.random() * 88 + "%";
        s.style.width = 14 + Math.random() * 16 + "px";
        s.style.animationDelay = Math.random() * 0.3 + "s";
        card.appendChild(s);
        setTimeout(() => s.remove(), 1600);
      }
    });
  });

  // 2) hero letters become individually poke-able once the intro settles
  document.querySelectorAll(".ltr").forEach((el) => {
    el.addEventListener("animationend", () => el.classList.add("done"), { once: true });
  });

  // 3) devtools peeper: cartoon eyes slide in and stare at the inspector
  if (!reduceMotion) {
    const peepEl = document.createElement("div");
    peepEl.className = "peep";
    peepEl.dataset.side = "right";
    peepEl.setAttribute("aria-hidden", "true");
    peepEl.innerHTML =
      `<div class="peep__eyes">` +
      `<span class="peep__eye"><span class="peep__pupil"></span></span>` +
      `<span class="peep__eye"><span class="peep__pupil"></span></span>` +
      `</div><span class="peep__tag">inspecting the inspector?</span>`;
    document.body.appendChild(peepEl);

    let peepTimer;
    let dockedOpen = false;
    const showPeep = (side) => {
      peepEl.dataset.side = side;
      peepEl.classList.add("on");
      clearTimeout(peepTimer);
      peepTimer = setTimeout(() => peepEl.classList.remove("on"), 8000);
    };

    // docked devtools: the window's outer/inner size delta gives the dock side
    setInterval(() => {
      const dRight = window.outerWidth - window.innerWidth > 170;
      const dBottom = window.outerHeight - window.innerHeight > 170;
      const open = dRight || dBottom;
      if (open && !dockedOpen) showPeep(dBottom && !dRight ? "bottom" : "right");
      if (!open && dockedOpen) peepEl.classList.remove("on");
      dockedOpen = open;
    }, 600);

    // undocked devtools: this image's id getter only runs when the console
    // actually renders the log entry, i.e. when someone opens it
    const bait = new Image();
    Object.defineProperty(bait, "id", {
      get() {
        if (!dockedOpen) showPeep("right");
        return "👀";
      },
    });
    console.log("%c", "", bait);
  }

  // 4) he's on the footer line. click him.
  const waldo = document.querySelector(".waldo");
  if (waldo) {
    waldo.addEventListener("click", () => {
      waldo.classList.add("found");
      console.log("%c( you found him )", "color:#e03131; font-family:monospace;");
      setTimeout(() => waldo.classList.remove("found"), 2600);
    });
  }

  // 5) a note for people who open the console
  console.log(
    "%cV–R" +
      "%c\n\nHey, you found the console." +
      "\nTry typing  w o l o l o  anywhere on the page." +
      "\n\nsreenivasan.ramesh@gmail.com",
    "font-size:40px; font-weight:bold; color:#ff4d24; font-family:Georgia,serif;",
    "font-size:12px; color:#8f8c83; font-family:monospace;"
  );

  /* ---------- custom cursor + magnetic elements (desktop only) ---------- */
  if (finePointer && !reduceMotion) {
    const dot = document.querySelector(".cursor");
    const ring = document.querySelector(".cursor-ring");
    const mouse = { x: -100, y: -100 };
    const ringPos = { x: -100, y: -100 };
    let cursorLive = false;

    window.addEventListener(
      "mousemove",
      (e) => {
        mouse.x = e.clientX;
        mouse.y = e.clientY;
        if (!cursorLive) {
          cursorLive = true;
          ringPos.x = mouse.x;
          ringPos.y = mouse.y;
          document.body.classList.add("cursor-on");
        }
      },
      { passive: true }
    );

    const hoverables = "a, button, .magnetic";
    document.addEventListener("mouseover", (e) => {
      if (e.target.closest(hoverables)) document.body.classList.add("cursor-hover");
    });
    document.addEventListener("mouseout", (e) => {
      if (e.target.closest(hoverables)) document.body.classList.remove("cursor-hover");
    });

    /* magnetic pull */
    const magnets = [...document.querySelectorAll(".magnetic")].map((el) => ({
      el,
      x: 0,
      y: 0,
      tx: 0,
      ty: 0,
    }));
    magnets.forEach((m) => {
      m.el.addEventListener("mousemove", (e) => {
        const r = m.el.getBoundingClientRect();
        m.tx = (e.clientX - (r.left + r.width / 2)) * 0.32;
        m.ty = (e.clientY - (r.top + r.height / 2)) * 0.32;
      });
      m.el.addEventListener("mouseleave", () => {
        m.tx = 0;
        m.ty = 0;
      });
    });

    const lerp = (a, b, t) => a + (b - a) * t;
    const peekEl = document.querySelector(".toss-peek");
    const peekPos = { x: -200, y: -200 };
    const loop = () => {
      dot.style.transform = `translate(${mouse.x}px, ${mouse.y}px)`;
      ringPos.x = lerp(ringPos.x, mouse.x, 0.16);
      ringPos.y = lerp(ringPos.y, mouse.y, 0.16);
      ring.style.transform = `translate(${ringPos.x}px, ${ringPos.y}px)`;
      peekPos.x = lerp(peekPos.x, mouse.x, 0.22);
      peekPos.y = lerp(peekPos.y, mouse.y, 0.22);
      peekEl.style.transform = `translate(${peekPos.x + 22}px, ${peekPos.y - 130}px)`;
      magnets.forEach((m) => {
        m.x = lerp(m.x, m.tx, 0.18);
        m.y = lerp(m.y, m.ty, 0.18);
        if (Math.abs(m.x) > 0.05 || Math.abs(m.y) > 0.05) {
          m.el.style.transform = `translate(${m.x}px, ${m.y}px)`;
        } else if (m.el.style.transform) {
          m.el.style.transform = "";
        }
      });
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }
})();
