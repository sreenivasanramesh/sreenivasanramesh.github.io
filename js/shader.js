/* Ambient flow-field background + wololo shockwave.
   Raw WebGL1, one fullscreen triangle, one fragment shader.
   Degrades to the plain CSS background when WebGL is unavailable
   or the visitor prefers reduced motion. */

(() => {
  "use strict";

  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  const canvas = document.createElement("canvas");
  canvas.className = "bg-shader";
  canvas.setAttribute("aria-hidden", "true");
  document.body.prepend(canvas);

  const gl =
    canvas.getContext("webgl", { antialias: false, alpha: false }) ||
    canvas.getContext("experimental-webgl", { antialias: false, alpha: false });
  if (!gl) {
    canvas.remove();
    return;
  }

  const VERT = `
    attribute vec2 a_pos;
    void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }
  `;

  const FRAG = `
    precision highp float;
    uniform vec2  u_res;
    uniform float u_time;
    uniform vec2  u_mouse;    // uv, y-up
    uniform vec3  u_accent;
    uniform float u_shockT;   // seconds since shock start, negative = idle
    uniform vec2  u_shockC;   // uv, y-up
    uniform float u_scrollV;  // smoothed scroll velocity, roughly -1..1
    uniform float u_scrollP;  // absolute scroll position in px

    float hash(vec2 p) {
      return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
    }
    float noise(vec2 p) {
      vec2 i = floor(p);
      vec2 f = fract(p);
      vec2 u = f * f * (3.0 - 2.0 * f);
      return mix(
        mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
        mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x),
        u.y
      );
    }
    float fbm(vec2 p) {
      float v = 0.0;
      float a = 0.5;
      for (int i = 0; i < 4; i++) {
        v += a * noise(p);
        p = p * 2.03 + vec2(11.7, 5.3);
        a *= 0.5;
      }
      return v;
    }

    void main() {
      vec2 uv = gl_FragCoord.xy / u_res;
      float aspect = u_res.x / u_res.y;
      vec2 p = uv * vec2(aspect, 1.0) * 1.7;
      float t = u_time * 0.028;
      // the field scrolls gently with the page and smears under fast scrolling
      p.y += u_scrollP * 0.00045;
      float smear = clamp(abs(u_scrollV), 0.0, 1.0);
      p.y = mix(p.y, p.y * 0.35 + u_scrollP * 0.00045 * 0.65, smear * 0.55);

      // expanding shock ring: displaces the field and glows in the accent
      float shock = 0.0;
      if (u_shockT >= 0.0) {
        vec2 d = uv - u_shockC;
        d.x *= aspect;
        float dist = length(d);
        float radius = u_shockT * 1.5;
        float band = exp(-pow((dist - radius) * 12.0, 2.0));
        float fade = 1.0 - smoothstep(0.0, 1.15, u_shockT);
        shock = band * fade;
        p += (d / max(dist, 1e-4)) * shock * 0.4;
      }

      // gentle pull toward the cursor
      vec2 md = uv - u_mouse;
      md.x *= aspect;
      float minfluence = exp(-dot(md, md) * 7.0);
      p -= md * minfluence * 0.18;

      // domain-warped fbm
      vec2 q = vec2(fbm(p + t), fbm(p + vec2(5.2, 1.3) - t));
      float n = fbm(p + 1.9 * q + vec2(t * 0.6, -t * 0.4));

      vec3 base = vec3(0.055, 0.055, 0.047);   // #0e0e0c
      vec3 soft = vec3(0.128, 0.126, 0.108);
      vec3 col = mix(base, soft, smoothstep(0.28, 0.85, n));
      col += u_accent * 0.06 * pow(max(q.y, 0.0), 2.5);
      // fast scrolling briefly charges the field with the accent
      col += u_accent * smear * 0.10 * smoothstep(0.4, 0.9, n);
      // the cursor is a soft light source
      col += vec3(0.95, 0.92, 0.85) * minfluence * 0.045;
      col += u_accent * minfluence * 0.02;
      col += u_accent * shock * 0.42;

      // soft vignette
      vec2 vc = uv - 0.5;
      col *= 1.0 - 0.45 * dot(vc, vc);

      gl_FragColor = vec4(col, 1.0);
    }
  `;

  const compile = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(s));
    }
    return s;
  };

  let program;
  try {
    program = gl.createProgram();
    gl.attachShader(program, compile(gl.VERTEX_SHADER, VERT));
    gl.attachShader(program, compile(gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(program));
    }
  } catch (e) {
    canvas.remove();
    return;
  }
  gl.useProgram(program);

  // fullscreen triangle
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(program, "a_pos");
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

  const U = {};
  ["u_res", "u_time", "u_mouse", "u_accent", "u_shockT", "u_shockC", "u_scrollV", "u_scrollP"].forEach(
    (n) => (U[n] = gl.getUniformLocation(program, n))
  );

  /* sizing: cap the pixel ratio, the field is soft anyway */
  const resize = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 1.25);
    canvas.width = Math.round(window.innerWidth * dpr);
    canvas.height = Math.round(window.innerHeight * dpr);
    gl.viewport(0, 0, canvas.width, canvas.height);
  };
  resize();
  window.addEventListener("resize", resize);

  /* accent color from the live CSS variable (wololo flips it) */
  const readAccent = () => {
    const hex = getComputedStyle(document.documentElement)
      .getPropertyValue("--accent")
      .trim();
    const m = /^#([0-9a-f]{6})$/i.exec(hex);
    if (!m) return;
    const v = parseInt(m[1], 16);
    gl.uniform3f(U.u_accent, ((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255);
  };
  readAccent();
  document.addEventListener("vr:accent", readAccent);

  /* shockwave trigger */
  let shockStart = -1;
  document.addEventListener("vr:shock", () => {
    shockStart = performance.now();
  });

  /* mouse, eased */
  const mouse = { x: 0.5, y: 0.45, tx: 0.5, ty: 0.45 };
  window.addEventListener(
    "mousemove",
    (e) => {
      mouse.tx = e.clientX / window.innerWidth;
      mouse.ty = 1.0 - e.clientY / window.innerHeight;
    },
    { passive: true }
  );

  gl.uniform2f(U.u_shockC, 0.5, 0.45);

  /* scroll state, smoothed */
  let scrollV = 0;
  let lastScrollY = window.scrollY;

  let raf;
  const frame = (now) => {
    mouse.x += (mouse.tx - mouse.x) * 0.04;
    mouse.y += (mouse.ty - mouse.y) * 0.04;
    const dy = window.scrollY - lastScrollY;
    lastScrollY = window.scrollY;
    scrollV += (Math.max(-1, Math.min(1, dy * 0.02)) - scrollV) * 0.08;
    gl.uniform1f(U.u_scrollV, scrollV);
    gl.uniform1f(U.u_scrollP, window.scrollY);
    gl.uniform2f(U.u_res, canvas.width, canvas.height);
    gl.uniform1f(U.u_time, now / 1000);
    gl.uniform2f(U.u_mouse, mouse.x, mouse.y);
    gl.uniform1f(U.u_shockT, shockStart < 0 ? -1 : (now - shockStart) / 1000);
    if (shockStart > 0 && now - shockStart > 1300) shockStart = -1;
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    raf = requestAnimationFrame(frame);
  };
  raf = requestAnimationFrame(frame);

  /* don't burn GPU in a hidden tab */
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      cancelAnimationFrame(raf);
    } else {
      raf = requestAnimationFrame(frame);
    }
  });

  /* tiny debug handle (used by automated checks) */
  window.__vrShader = {
    shock: () => document.dispatchEvent(new CustomEvent("vr:shock")),
    freezeShock: (t) => { shockStart = performance.now() - t * 1000; },
  };
})();
