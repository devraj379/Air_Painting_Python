(() => {
  const WIDTH = 640;
  const HEIGHT = 480;

  // MediaPipe 21-landmark hand connections (matches mp.solutions.hands.HAND_CONNECTIONS)
  const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20],
    [0, 17]
  ];

  // Top air-hover zones (inspired by AIR_PAINTING.py y <= 65)
  const AIR_BUTTONS = [
    { id: 'clear', label: 'CLEAR', x1: 16, x2: 110, color: '#ef4444', dwellMs: 400 },
    { id: 'blue', label: 'BLUE', x1: 120, x2: 214, color: '#2563eb', dwellMs: 120 },
    { id: 'green', label: 'GREEN', x1: 224, x2: 318, color: '#16a34a', dwellMs: 120 },
    { id: 'red', label: 'RED', x1: 328, x2: 422, color: '#dc2626', dwellMs: 120 },
    { id: 'yellow', label: 'YELLOW', x1: 432, x2: 526, color: '#eab308', dwellMs: 120 },
    { id: 'eraser', label: 'ERASER', x1: 536, x2: 624, color: '#94a3b8', dwellMs: 180 }
  ];

  // DOM Elements
  const stage = document.getElementById('stage');
  const video = document.getElementById('webcamVideo');
  const cameraCanvas = document.getElementById('cameraCanvas');
  const cameraCtx = cameraCanvas.getContext('2d');
  const arBgCanvas = document.getElementById('arBgCanvas');
  const arBgCtx = arBgCanvas.getContext('2d');
  const paintCanvas = document.getElementById('paintCanvas');
  const paintCtx = paintCanvas.getContext('2d');
  const cursorCanvas = document.getElementById('cursorCanvas');
  const cursorCtx = cursorCanvas.getContext('2d');
  const cameraPlaceholder = document.getElementById('cameraPlaceholder');

  const toggleCameraBtn = document.getElementById('toggleCameraBtn');
  const cameraBtnIcon = document.getElementById('cameraBtnIcon');
  const cameraBtnLabel = document.getElementById('cameraBtnLabel');
  const connectionBadge = document.getElementById('connectionBadge');
  const connectionText = document.getElementById('connectionText');
  const gestureBadge = document.getElementById('gestureBadge');
  const gestureText = document.getElementById('gestureText');
  const fpsValue = document.getElementById('fpsValue');
  const deltaYValue = document.getElementById('deltaYValue');

  const swatchButtons = Array.from(document.querySelectorAll('.swatch'));
  const customColorPicker = document.getElementById('customColorPicker');
  const eraserBtn = document.getElementById('eraserBtn');
  const brushSizeInput = document.getElementById('brushSize');
  const brushSizeVal = document.getElementById('brushSizeVal');
  const undoBtn = document.getElementById('undoBtn');
  const clearBtn = document.getElementById('clearBtn');
  const downloadBtn = document.getElementById('downloadBtn');
  const beautifyBtn = document.getElementById('beautifyBtn');
  const autoShapeToggle = document.getElementById('autoShapeToggle');
  const aiGuessBtn = document.getElementById('aiGuessBtn');
  const aiGuessText = document.getElementById('aiGuessText');
  const aiConfidenceText = document.getElementById('aiConfidenceText');
  const aiTipText = document.getElementById('aiTipText');
  const challengeTarget = document.getElementById('challengeTarget');
  const challengeStatus = document.getElementById('challengeStatus');
  const nextChallengeBtn = document.getElementById('nextChallengeBtn');

  const segButtons = Array.from(document.querySelectorAll('.seg-btn'));
  const showSkeletonToggle = document.getElementById('showSkeleton');
  const showAirBarToggle = document.getElementById('showAirBar');
  const darkCanvasToggle = document.getElementById('darkCanvasToggle');
  const pinchThresholdInput = document.getElementById('pinchThreshold');
  const pinchThresholdVal = document.getElementById('pinchThresholdVal');
  const smoothToggle = document.getElementById('smoothToggle');
  const mouseDrawToggle = document.getElementById('mouseDrawToggle');

  // Offscreen canvas for sending mirrored low-latency JPEG frames to backend
  const CAPTURE_WIDTH = 320;
  const CAPTURE_HEIGHT = 240;
  const captureCanvas = document.createElement('canvas');
  captureCanvas.width = CAPTURE_WIDTH;
  captureCanvas.height = CAPTURE_HEIGHT;
  const captureCtx = captureCanvas.getContext('2d', { alpha: false });

  // Offscreen cache for committed strokes so per-frame rendering stays O(1)
  const committedCanvas = document.createElement('canvas');
  committedCanvas.width = WIDTH;
  committedCanvas.height = HEIGHT;
  const committedCtx = committedCanvas.getContext('2d');

  // Application State
  const state = {
    viewMode: 'split', // 'split' | 'overlay' | 'canvas'
    activeColor: '#2563eb',
    isEraser: false,
    brushSize: 6,
    pinchThreshold: 30,
    strokes: [],
    currentAirStroke: null,
    currentMouseStroke: null,
    cameraRunning: false,
    stream: null,
    ws: null,
    awaitingResponse: false,
    lastSendTime: 0,
    latestTracking: null,
    smoothedPoint: null,
    hoverButtonId: null,
    hoverStartTime: 0,
    hoverTriggered: false,
    fpsFrames: 0,
    fpsLastUpdate: performance.now(),
    rafId: null,
    aiChallenges: ['House', 'Smiley Face', 'Star', 'Heart', 'Sun', 'Triangle', 'Circle', 'Square'],
    challengeIndex: 0,
    analyzeTimer: null
  };

  // --- Painting Engine ---
  function rebuildCommittedCanvas() {
    committedCtx.clearRect(0, 0, WIDTH, HEIGHT);
    for (const stroke of state.strokes) {
      drawSingleStroke(committedCtx, stroke);
    }
  }

  function renderStrokes() {
    paintCtx.clearRect(0, 0, WIDTH, HEIGHT);
    paintCtx.drawImage(committedCanvas, 0, 0);
    if (state.currentAirStroke) drawSingleStroke(paintCtx, state.currentAirStroke);
    if (state.currentMouseStroke) drawSingleStroke(paintCtx, state.currentMouseStroke);
  }

  function drawSingleStroke(ctx, stroke) {
    const pts = stroke.points;
    if (!pts || pts.length === 0) return;

    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = stroke.size;
    if (stroke.isEraser) {
      ctx.globalCompositeOperation = 'destination-out';
      ctx.strokeStyle = 'rgba(0,0,0,1)';
      ctx.fillStyle = 'rgba(0,0,0,1)';
    } else {
      ctx.globalCompositeOperation = 'source-over';
      ctx.strokeStyle = stroke.color;
      ctx.fillStyle = stroke.color;
    }

    if (pts.length === 1) {
      ctx.beginPath();
      ctx.arc(pts[0].x, pts[0].y, stroke.size / 2, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
      return;
    }

    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length - 1; i++) {
      const midX = (pts[i].x + pts[i + 1].x) / 2;
      const midY = (pts[i].y + pts[i + 1].y) / 2;
      ctx.quadraticCurveTo(pts[i].x, pts[i].y, midX, midY);
    }
    const last = pts[pts.length - 1];
    ctx.lineTo(last.x, last.y);
    ctx.stroke();
    ctx.restore();
  }

  function finishAirStroke() {
    if (state.currentAirStroke && state.currentAirStroke.points.length > 0) {
      const stroke = state.currentAirStroke;
      state.strokes.push(stroke);
      drawSingleStroke(committedCtx, stroke);
      onStrokeCompleted(stroke);
    }
    state.currentAirStroke = null;
    renderStrokes();
  }

  function clearCanvas() {
    state.strokes = [];
    state.currentAirStroke = null;
    state.currentMouseStroke = null;
    rebuildCommittedCanvas();
    renderStrokes();
    scheduleAIAnalysis(50);
  }

  function undoLastStroke() {
    if (state.strokes.length > 0) {
      state.strokes.pop();
      rebuildCommittedCanvas();
      renderStrokes();
      scheduleAIAnalysis(50);
    }
  }

  // --- AI Shape Beautifier & Doodle Guesser ---
  function onStrokeCompleted(stroke) {
    if (autoShapeToggle.checked && !stroke.isEraser && stroke.points.length >= 4) {
      beautifyStrokeObject(stroke, true);
    } else {
      scheduleAIAnalysis(280);
    }
  }

  async function beautifyStrokeObject(targetStroke, autoOnly) {
    if (!targetStroke || targetStroke.isEraser || targetStroke.points.length < 2) return;
    try {
      const res = await fetch('/api/ai/beautify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ points: targetStroke.points })
      });
      if (!res.ok) return;
      const data = await res.json();
      const isGeometric = data.shape && data.shape !== 'Freehand' && data.shape !== 'Dot';
      if (autoOnly && (!isGeometric || data.confidence < 0.72)) {
        scheduleAIAnalysis(250);
        return;
      }
      if (Array.isArray(data.beautified_points) && data.beautified_points.length >= 2) {
        targetStroke.points = data.beautified_points;
        rebuildCommittedCanvas();
        renderStrokes();
        aiGuessText.textContent = `Snapped: ${data.shape}`;
        aiConfidenceText.textContent = `${Math.round((data.confidence || 0) * 100)}%`;
      }
    } catch (err) {
      console.error('AI beautify error:', err);
    } finally {
      scheduleAIAnalysis(200);
    }
  }

  function beautifyLastStroke() {
    for (let i = state.strokes.length - 1; i >= 0; i--) {
      if (!state.strokes[i].isEraser && state.strokes[i].points.length >= 2) {
        beautifyStrokeObject(state.strokes[i], false);
        return;
      }
    }
  }

  function scheduleAIAnalysis(delayMs = 250) {
    if (state.analyzeTimer) clearTimeout(state.analyzeTimer);
    state.analyzeTimer = setTimeout(() => {
      analyzeCanvasWithAI();
    }, delayMs);
  }

  async function analyzeCanvasWithAI() {
    const activeChallenge = state.aiChallenges[state.challengeIndex];
    try {
      const res = await fetch('/api/ai/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          strokes: state.strokes,
          target_challenge: activeChallenge
        })
      });
      if (!res.ok) return;
      const data = await res.json();
      aiGuessText.textContent = data.guess || 'Unknown';
      aiConfidenceText.textContent = data.guess === 'Blank Canvas' ? '' : `${data.confidence}%`;
      aiTipText.textContent = data.tip || '';

      if (data.challenge_match) {
        challengeStatus.textContent = 'Matched!';
        challengeStatus.classList.add('matched');
      } else {
        challengeStatus.textContent = 'In progress';
        challengeStatus.classList.remove('matched');
      }
    } catch (err) {
      console.error('AI analyze error:', err);
    }
  }

  function selectColor(color) {
    state.activeColor = color;
    state.isEraser = false;
    eraserBtn.classList.remove('active');
    swatchButtons.forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.color.toLowerCase() === color.toLowerCase());
    });
  }

  function toggleEraser(forceValue) {
    state.isEraser = typeof forceValue === 'boolean' ? forceValue : !state.isEraser;
    eraserBtn.classList.toggle('active', state.isEraser);
    if (state.isEraser) {
      swatchButtons.forEach((btn) => btn.classList.remove('active'));
    }
  }

  // --- Badge Helpers ---
  function setConnectionStatus(kind, text) {
    connectionBadge.className = `badge ${kind}`;
    connectionText.textContent = text;
  }

  function setGestureStatus(kind, text) {
    gestureBadge.className = `badge ${kind}`;
    gestureText.textContent = text;
  }

  // --- WebSocket Connection ---
  function connectWebSocket() {
    if (state.ws && (state.ws.readyState === WebSocket.OPEN || state.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/track`;
    setConnectionStatus('badge-warn', 'Connecting…');

    const ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';
    state.ws = ws;

    ws.onopen = () => {
      state.awaitingResponse = false;
      setConnectionStatus('badge-ok', 'Python MediaPipe Ready');
      sendTrackingFrame();
    };

    ws.onmessage = (event) => {
      state.awaitingResponse = false;
      try {
        const data = JSON.parse(event.data);
        handleTrackingResult(data);
      } catch (err) {
        console.error('Invalid tracking payload:', err);
      }
      if (state.cameraRunning) {
        sendTrackingFrame();
      }
    };

    ws.onerror = () => {
      state.awaitingResponse = false;
      setConnectionStatus('badge-err', 'Tracking Error');
    };

    ws.onclose = () => {
      state.awaitingResponse = false;
      setConnectionStatus('badge-err', 'Disconnected');
      if (state.cameraRunning) {
        setTimeout(connectWebSocket, 1500);
      }
    };
  }

  function disconnectWebSocket() {
    if (state.ws) {
      const ws = state.ws;
      state.ws = null;
      ws.close();
    }
    state.awaitingResponse = false;
  }

  // --- Air Tracking Logic ---
  function handleTrackingResult(data) {
    state.latestTracking = data;

    // Update FPS counter
    state.fpsFrames += 1;
    const now = performance.now();
    if (now - state.fpsLastUpdate >= 500) {
      const fps = Math.round((state.fpsFrames * 1000) / (now - state.fpsLastUpdate));
      fpsValue.textContent = String(fps);
      state.fpsFrames = 0;
      state.fpsLastUpdate = now;
    }

    if (!data || !data.detected) {
      deltaYValue.textContent = '—';
      state.smoothedPoint = null;
      resetHoverState();
      finishAirStroke();
      if (state.cameraRunning) {
        setGestureStatus('badge-neutral', 'No Hand Detected');
      }
      return;
    }

    const rawIndex = data.indexTip;
    const rawThumb = data.thumbTip;
    const deltaY = Math.round(rawThumb[1] - rawIndex[1]);
    deltaYValue.textContent = `${deltaY}px`;

    // Smooth index tip adaptively (snappy on fast moves, steady on fine detail)
    let pt;
    if (smoothToggle.checked && state.smoothedPoint) {
      const speed = Math.hypot(rawIndex[0] - state.smoothedPoint.x, rawIndex[1] - state.smoothedPoint.y);
      const alpha = speed > 18 ? 0.78 : 0.48;
      pt = {
        x: state.smoothedPoint.x + alpha * (rawIndex[0] - state.smoothedPoint.x),
        y: state.smoothedPoint.y + alpha * (rawIndex[1] - state.smoothedPoint.y)
      };
    } else {
      pt = { x: rawIndex[0], y: rawIndex[1] };
    }
    state.smoothedPoint = pt;

    // 1. Pinch Lift Check (mirrors thumb[1] - center[1] < 30 in AIR_PAINTING.py)
    if (deltaY < state.pinchThreshold) {
      finishAirStroke();
      resetHoverState();
      setGestureStatus('badge-warn', 'Pen Up (Pinched)');
      return;
    }

    // 2. Top Air-Toolbar Zone (y <= 65)
    if (showAirBarToggle.checked && pt.y <= 65) {
      finishAirStroke();
      handleAirToolbarHover(pt.x, now);
      setGestureStatus('badge-neutral', 'Selecting Tool');
      return;
    }

    resetHoverState();

    // 3. Draw on Canvas
    setGestureStatus('badge-draw', state.isEraser ? 'Erasing' : 'Drawing');
    if (!state.currentAirStroke) {
      state.currentAirStroke = {
        color: state.activeColor,
        size: state.brushSize,
        isEraser: state.isEraser,
        points: [pt]
      };
    } else {
      const pts = state.currentAirStroke.points;
      const prev = pts[pts.length - 1];
      const dist = Math.hypot(pt.x - prev.x, pt.y - prev.y);
      if (dist >= 1.5) {
        pts.push(pt);
      }
    }
    renderStrokes();
  }

  function resetHoverState() {
    state.hoverButtonId = null;
    state.hoverStartTime = 0;
    state.hoverTriggered = false;
  }

  function handleAirToolbarHover(x, now) {
    const btn = AIR_BUTTONS.find((b) => x >= b.x1 && x <= b.x2);
    if (!btn) {
      resetHoverState();
      return;
    }

    if (state.hoverButtonId !== btn.id) {
      state.hoverButtonId = btn.id;
      state.hoverStartTime = now;
      state.hoverTriggered = false;
      return;
    }

    if (!state.hoverTriggered && now - state.hoverStartTime >= btn.dwellMs) {
      state.hoverTriggered = true;
      if (btn.id === 'clear') {
        clearCanvas();
      } else if (btn.id === 'eraser') {
        toggleEraser(true);
      } else {
        selectColor(btn.color);
      }
    }
  }

  // --- Camera & Render Loop ---
  async function startCamera() {
    if (state.cameraRunning) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false
      });
      state.stream = stream;
      video.srcObject = stream;
      await video.play();

      state.cameraRunning = true;
      cameraPlaceholder.classList.add('hidden');
      cameraBtnIcon.textContent = '■';
      cameraBtnLabel.textContent = 'Stop Camera';
      toggleCameraBtn.classList.remove('btn-primary');
      toggleCameraBtn.classList.add('btn-danger-ghost');
      setGestureStatus('badge-neutral', 'Looking for Hand…');

      connectWebSocket();
      loop();
    } catch (err) {
      console.error('Camera error:', err);
      setGestureStatus('badge-err', 'Camera Permission Denied');
      alert('Could not access your webcam. Please allow camera permissions and try again.');
    }
  }

  function stopCamera() {
    state.cameraRunning = false;
    if (state.rafId) {
      cancelAnimationFrame(state.rafId);
      state.rafId = null;
    }
    finishAirStroke();
    if (state.stream) {
      state.stream.getTracks().forEach((t) => t.stop());
      state.stream = null;
    }
    video.srcObject = null;
    state.latestTracking = null;
    state.smoothedPoint = null;
    resetHoverState();
    cameraCtx.clearRect(0, 0, WIDTH, HEIGHT);
    arBgCtx.clearRect(0, 0, WIDTH, HEIGHT);
    cursorCtx.clearRect(0, 0, WIDTH, HEIGHT);
    cameraPlaceholder.classList.remove('hidden');

    cameraBtnIcon.textContent = '▶';
    cameraBtnLabel.textContent = 'Start Camera';
    toggleCameraBtn.classList.add('btn-primary');
    toggleCameraBtn.classList.remove('btn-danger-ghost');
    fpsValue.textContent = '0';
    deltaYValue.textContent = '—';
    setGestureStatus('badge-neutral', 'Camera Off');
    disconnectWebSocket();
  }

  function drawCameraOverlay(ctx, includeStrokes) {
    // Draw mirrored video frame
    ctx.save();
    ctx.translate(WIDTH, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, WIDTH, HEIGHT);
    ctx.restore();

    if (includeStrokes) {
      ctx.drawImage(paintCanvas, 0, 0, WIDTH, HEIGHT);
    }

    if (showAirBarToggle.checked) {
      drawAirToolbar(ctx);
    }

    if (showSkeletonToggle.checked && state.latestTracking && state.latestTracking.detected) {
      drawHandSkeleton(ctx, state.latestTracking);
    }
  }

  function drawAirToolbar(ctx) {
    const now = performance.now();
    for (const btn of AIR_BUTTONS) {
      const isHovered = state.hoverButtonId === btn.id;
      const isActiveColor =
        (!state.isEraser && btn.color.toLowerCase() === state.activeColor.toLowerCase() && btn.id !== 'clear') ||
        (state.isEraser && btn.id === 'eraser');

      ctx.save();
      ctx.fillStyle = isHovered ? 'rgba(15, 23, 42, 0.85)' : 'rgba(15, 23, 42, 0.62)';
      ctx.strokeStyle = isActiveColor ? '#ffffff' : btn.color;
      ctx.lineWidth = isActiveColor ? 3 : 2;
      roundRect(ctx, btn.x1, 8, btn.x2 - btn.x1, 50, 8);
      ctx.fill();
      ctx.stroke();

      // Dwell progress bar for hover feedback
      if (isHovered && !state.hoverTriggered && state.hoverStartTime > 0) {
        const progress = Math.min(1, (now - state.hoverStartTime) / btn.dwellMs);
        ctx.fillStyle = btn.color;
        ctx.fillRect(btn.x1 + 3, 52, (btn.x2 - btn.x1 - 6) * progress, 4);
      }

      ctx.fillStyle = '#f8fafc';
      ctx.font = 'bold 12px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(btn.label, (btn.x1 + btn.x2) / 2, 33);
      ctx.restore();
    }
  }

  function drawHandSkeleton(ctx, tracking) {
    const lms = tracking.landmarks;
    if (!lms || lms.length < 21) return;

    ctx.save();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.72)';
    ctx.lineWidth = 2;
    for (const [a, b] of HAND_CONNECTIONS) {
      ctx.beginPath();
      ctx.moveTo(lms[a][0], lms[a][1]);
      ctx.lineTo(lms[b][0], lms[b][1]);
      ctx.stroke();
    }

    for (let i = 0; i < lms.length; i++) {
      ctx.beginPath();
      ctx.arc(lms[i][0], lms[i][1], i === 8 || i === 4 ? 5 : 3, 0, Math.PI * 2);
      ctx.fillStyle = i === 8 ? '#22c55e' : i === 4 ? '#f59e0b' : '#38bdf8';
      ctx.fill();
    }
    ctx.restore();
  }

  function drawPaintCursor() {
    cursorCtx.clearRect(0, 0, WIDTH, HEIGHT);
    if (!state.cameraRunning || !state.smoothedPoint) return;

    const pt = state.smoothedPoint;
    const tracking = state.latestTracking;
    if (!tracking || !tracking.detected) return;

    const deltaY = tracking.thumbTip[1] - tracking.indexTip[1];
    const isPinched = deltaY < state.pinchThreshold;
    const isSelecting = showAirBarToggle.checked && pt.y <= 65;

    cursorCtx.save();
    cursorCtx.beginPath();
    cursorCtx.arc(pt.x, pt.y, Math.max(4, state.brushSize / 2), 0, Math.PI * 2);
    if (isPinched || isSelecting) {
      cursorCtx.strokeStyle = '#f59e0b';
      cursorCtx.lineWidth = 2;
      cursorCtx.setLineDash([4, 3]);
      cursorCtx.stroke();
    } else {
      cursorCtx.strokeStyle = state.isEraser ? '#94a3b8' : state.activeColor;
      cursorCtx.lineWidth = 2.5;
      cursorCtx.stroke();
      cursorCtx.fillStyle = state.isEraser ? 'rgba(148,163,184,0.25)' : state.activeColor;
      cursorCtx.globalAlpha = 0.35;
      cursorCtx.fill();
    }
    cursorCtx.restore();
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  function sendTrackingFrame() {
    if (
      !state.cameraRunning ||
      video.readyState < 2 ||
      !state.ws ||
      state.ws.readyState !== WebSocket.OPEN
    ) {
      return;
    }
    const now = performance.now();
    if (state.awaitingResponse && now - state.lastSendTime < 1500) {
      return;
    }

    state.awaitingResponse = true;
    state.lastSendTime = now;

    captureCtx.save();
    captureCtx.translate(CAPTURE_WIDTH, 0);
    captureCtx.scale(-1, 1);
    captureCtx.drawImage(video, 0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT);
    captureCtx.restore();

    captureCanvas.toBlob(
      (blob) => {
        if (blob && state.ws && state.ws.readyState === WebSocket.OPEN) {
          state.ws.send(blob);
        } else {
          state.awaitingResponse = false;
        }
      },
      'image/jpeg',
      0.55
    );
  }

  function loop() {
    if (!state.cameraRunning) return;

    if (video.readyState >= 2) {
      if (state.viewMode === 'split') {
        drawCameraOverlay(cameraCtx, true);
      } else if (state.viewMode === 'overlay') {
        drawCameraOverlay(arBgCtx, false);
      }
      drawPaintCursor();

      if (!state.awaitingResponse) {
        sendTrackingFrame();
      }
    }

    state.rafId = requestAnimationFrame(loop);
  }

  // --- Mouse / Touch Fallback Drawing ---
  function getCanvasCoords(e) {
    const rect = paintCanvas.getBoundingClientRect();
    const scaleX = WIDTH / rect.width;
    const scaleY = HEIGHT / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    };
  }

  paintCanvas.addEventListener('pointerdown', (e) => {
    if (!mouseDrawToggle.checked) return;
    paintCanvas.setPointerCapture(e.pointerId);
    const pt = getCanvasCoords(e);
    state.currentMouseStroke = {
      color: state.activeColor,
      size: state.brushSize,
      isEraser: state.isEraser,
      points: [pt]
    };
    renderStrokes();
  });

  paintCanvas.addEventListener('pointermove', (e) => {
    if (!state.currentMouseStroke) return;
    const pt = getCanvasCoords(e);
    state.currentMouseStroke.points.push(pt);
    renderStrokes();
  });

  function endMouseStroke() {
    if (state.currentMouseStroke && state.currentMouseStroke.points.length > 0) {
      const stroke = state.currentMouseStroke;
      state.strokes.push(stroke);
      drawSingleStroke(committedCtx, stroke);
      onStrokeCompleted(stroke);
    }
    state.currentMouseStroke = null;
    renderStrokes();
  }

  paintCanvas.addEventListener('pointerup', endMouseStroke);
  paintCanvas.addEventListener('pointercancel', endMouseStroke);

  // --- Export PNG ---
  function downloadPng() {
    const exportCanvas = document.createElement('canvas');
    exportCanvas.width = WIDTH;
    exportCanvas.height = HEIGHT;
    const ctx = exportCanvas.getContext('2d');

    if (state.viewMode === 'overlay' && state.cameraRunning) {
      ctx.drawImage(arBgCanvas, 0, 0, WIDTH, HEIGHT);
    } else {
      ctx.fillStyle = darkCanvasToggle.checked ? '#0f172a' : '#ffffff';
      ctx.fillRect(0, 0, WIDTH, HEIGHT);
    }
    ctx.drawImage(paintCanvas, 0, 0, WIDTH, HEIGHT);

    const link = document.createElement('a');
    link.download = `air-painting-${Date.now()}.png`;
    link.href = exportCanvas.toDataURL('image/png');
    link.click();
  }

  // --- UI Event Bindings ---
  toggleCameraBtn.addEventListener('click', () => {
    if (state.cameraRunning) {
      stopCamera();
    } else {
      startCamera();
    }
  });

  segButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      state.viewMode = btn.dataset.view;
      segButtons.forEach((b) => b.classList.toggle('active', b === btn));
      stage.className = `stage mode-${state.viewMode}`;
    });
  });

  swatchButtons.forEach((btn) => {
    btn.addEventListener('click', () => selectColor(btn.dataset.color));
  });

  customColorPicker.addEventListener('input', (e) => {
    selectColor(e.target.value);
  });

  eraserBtn.addEventListener('click', () => toggleEraser());

  brushSizeInput.addEventListener('input', (e) => {
    state.brushSize = Number(e.target.value);
    brushSizeVal.textContent = String(state.brushSize);
  });

  pinchThresholdInput.addEventListener('input', (e) => {
    state.pinchThreshold = Number(e.target.value);
    pinchThresholdVal.textContent = String(state.pinchThreshold);
  });

  darkCanvasToggle.addEventListener('change', (e) => {
    paintCanvas.classList.toggle('dark-bg', e.target.checked);
  });

  undoBtn.addEventListener('click', undoLastStroke);
  clearBtn.addEventListener('click', clearCanvas);
  downloadBtn.addEventListener('click', downloadPng);
  beautifyBtn.addEventListener('click', beautifyLastStroke);
  aiGuessBtn.addEventListener('click', analyzeCanvasWithAI);
  nextChallengeBtn.addEventListener('click', () => {
    state.challengeIndex = (state.challengeIndex + 1) % state.aiChallenges.length;
    challengeTarget.textContent = state.aiChallenges[state.challengeIndex];
    analyzeCanvasWithAI();
  });

  window.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
      e.preventDefault();
      undoLastStroke();
    }
  });

  // Check backend health on initial page load
  fetch('/api/health')
    .then((r) => r.json())
    .then((info) => {
      if (info && info.status === 'ok') {
        setConnectionStatus('badge-ok', 'Backend Online');
      } else {
        setConnectionStatus('badge-warn', 'Backend Degraded');
      }
    })
    .catch(() => {
      setConnectionStatus('badge-err', 'Backend Offline');
    });
})();
