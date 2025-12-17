window.AIG = (function(){
  let analyzeMode = 'detect'; // 'detect' | 'disease'

  // ---- Router ----
  function setActiveTab(page){
    const backdrop = document.getElementById('page-backdrop');
    // Hide all pages first
    document.querySelectorAll('.page').forEach(p => p.classList.add('hidden'));
    // If explicitly switching to chat, hide backdrop and return
    if (page === 'chat') {
      if (backdrop) backdrop.classList.add('hidden');
      return;
    }
    // Show requested page and backdrop
    const el = document.getElementById(`page-${page}`);
    if (el) {
      el.classList.remove('hidden');
      if (backdrop) backdrop.classList.remove('hidden');
    }
  }

  function bindTabs(){
    document.querySelectorAll('.tab').forEach(btn => {
      btn.addEventListener('click', () => setActiveTab(btn.dataset.page));
    });
  }

  // ---- Search ----
  let q, runBtn, statusEl, resultsEl;

  function setStatus(msg, busy=false){
    statusEl.textContent = msg;
    statusEl.style.opacity = 1;
    if (runBtn) runBtn.disabled = busy;
  }

  function renderResults(items){
    resultsEl.innerHTML = '';
    if(!items || items.length === 0){
      const li = document.createElement('li');
      li.textContent = 'No results.';
      resultsEl.appendChild(li);
      return;
    }
    for (const r of items){
      const li = document.createElement('li');
      const sim = typeof r.similarity === 'number' ? r.similarity.toFixed(3) : 'n/a';
      li.innerHTML = `<div class="sim">similarity: ${sim}</div><div>${(r.content || '').replace(/</g,'&lt;')}</div>`;
      resultsEl.appendChild(li);
    }
  }

  async function search(){
    const query = q.value.trim();
    if(!query){
      setStatus('Please enter a query.');
      return;
    }
    setStatus('Searching…', true);
    try {
      const resp = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, k: 5 })
      });
      if(!resp.ok){
        const text = await resp.text();
        throw new Error(`HTTP ${resp.status}: ${text}`);
      }
      const data = await resp.json();
      renderResults(data.results || []);
      setStatus('Done.', false);
    } catch (err){
      console.error(err);
      setStatus('Error: ' + err.message, false);
    }
  }

  function initSearch(){
    q = document.getElementById('query');
    runBtn = document.getElementById('btnSearch');
    statusEl = document.getElementById('status');
    resultsEl = document.getElementById('results');
    runBtn?.addEventListener('click', search);
    q?.addEventListener('keydown', (e) => { if(e.key === 'Enter'){ search(); }});
  }

  // ---- Analyze ----
  function initAnalyze(){
    const fileInput = document.getElementById('an-image');
    const preview = document.getElementById('an-preview');
    const run = document.getElementById('an-run');
    const out = document.getElementById('an-results');

    fileInput?.addEventListener('change', () => {
      preview.innerHTML = '';
      const f = fileInput.files?.[0];
      if (!f) return;
      const img = document.createElement('img');
      img.src = URL.createObjectURL(f);
      preview.appendChild(img);
    });

    run?.addEventListener('click', async () => {
      out.innerHTML = '';
      const f = fileInput?.files?.[0];
      if (!f && !mockMode){
        out.appendChild(card('Error', 'Please add an image before diagnosing.'));
        return;
      }
      try {
        if (analyzeMode === 'detect'){
          const fd = new FormData();
          if (f) fd.append('image', f, f.name);
          const resp = await fetch('/api/detect', { method: 'POST', body: fd });
          if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
          const det = await resp.json();
          const cropLabel = det.output?.label ?? 'n/a';
          const cropConf = det.output?.confidence ?? 'n/a';
          out.appendChild(card('Crop detection (crop_detection)', `label=${cropLabel}, confidence=${cropConf}`));
          if (Array.isArray(det.output?.top3)){
            const lines = det.output.top3.map(t => `${t.label}: ${t.confidence}`).join('\n');
            out.appendChild(card('Top-3', lines));
          }
          if (det.output?.note){ out.appendChild(card('Note', det.output.note)); }
          const top3line = Array.isArray(det.output?.top3) ?
            'Top-3: ' + det.output.top3.map(t => `${t.label}(${t.confidence})`).join(', ') :
            undefined;
          const summary = [
            'Crop detection (crop_detection)',
            `Input: filename="${det.input?.filename || (f?.name||'')}"`,
            `Output: crop=${cropLabel}, confidence=${cropConf}`,
            top3line
          ].filter(Boolean).join('\n');
          appendChatBot(summary);
        } else {
          const fd = new FormData();
          if (f) fd.append('image', f, f.name);
          const resp = await fetch('/api/disease', { method:'POST', body: fd });
          if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
          const dis = await resp.json();
          const label = dis.output?.label ?? 'n/a';
          const conf = dis.output?.confidence ?? 'n/a';
          out.appendChild(card('Disease detection', `label=${label}, confidence=${conf}`));
          if (Array.isArray(dis.output?.top3)){
            const lines = dis.output.top3.map(t => `${t.label}: ${t.confidence}`).join('\n');
            out.appendChild(card('Top-3', lines));
          }
          if (dis.output?.note){ out.appendChild(card('Note', dis.output.note)); }
          const top3line = Array.isArray(dis.output?.top3) ?
            'Top-3: ' + dis.output.top3.map(t => `${t.label}(${t.confidence})`).join(', ') :
            undefined;
          const summary = [
            'Disease detection',
            `Input: filename="${dis.input?.filename || (f?.name||'')}"`,
            `Output: disease=${label}, confidence=${conf}`,
            top3line
          ].filter(Boolean).join('\n');
          appendChatBot(summary);
        }
      } catch (err){
        out.appendChild(card('Error', String(err.message||err)));
      }
    });

    // Update UI for crop vs disease mode
    updateAnalyzeUI();
  }

  // ---- Companions ----
  function initCompanions(){
    const plant = document.getElementById('cp-plant');
    const btn = document.getElementById('cp-suggest');
    const box = document.getElementById('cp-suggestions');
    btn?.addEventListener('click', async () => {
      box.innerHTML = '';
      const v = (plant.value || '').trim();
      // Creates a two-column layout with titles and counts
      function renderNice(goodArr, badArr, infoText){
        box.innerHTML = '';
        if (infoText){ const info = document.createElement('div'); info.className = 'cp-info'; info.textContent = infoText; box.appendChild(info); }
        const grid = document.createElement('div'); grid.className = 'cp-results'; box.appendChild(grid);
        // Helps column
        const colGood = document.createElement('div'); colGood.className = 'cp-col'; grid.appendChild(colGood);
        const headG = document.createElement('div'); headG.className = 'cp-header'; colGood.appendChild(headG);
        const ttlG = document.createElement('div'); ttlG.className = 'cp-title'; ttlG.textContent = 'Helps'; headG.appendChild(ttlG);
        const cntG = document.createElement('div'); cntG.className = 'cp-count'; cntG.textContent = `${(goodArr||[]).length}`; headG.appendChild(cntG);
        if (goodArr && goodArr.length){
          const tags = document.createElement('div'); tags.className = 'tags'; colGood.appendChild(tags);
          for (const it of goodArr){ tags.appendChild(tag(it, 'good')); }
        } else {
          const empty = document.createElement('div'); empty.className = 'cp-empty'; empty.textContent = 'No helpful companions found.'; colGood.appendChild(empty);
        }
        // Avoid column
        const colBad = document.createElement('div'); colBad.className = 'cp-col'; grid.appendChild(colBad);
        const headB = document.createElement('div'); headB.className = 'cp-header'; colBad.appendChild(headB);
        const ttlB = document.createElement('div'); ttlB.className = 'cp-title'; ttlB.textContent = 'Avoid'; headB.appendChild(ttlB);
        const cntB = document.createElement('div'); cntB.className = 'cp-count'; cntB.textContent = `${(badArr||[]).length}`; headB.appendChild(cntB);
        if (badArr && badArr.length){
          const tags = document.createElement('div'); tags.className = 'tags'; colBad.appendChild(tags);
          for (const it of badArr){ tags.appendChild(tag(it, 'bad')); }
        } else {
          const empty = document.createElement('div'); empty.className = 'cp-empty'; empty.textContent = 'No conflicts found.'; colBad.appendChild(empty);
        }
      }

      try {
        const resp = await fetch('/api/companions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ plant: v })
        });
        if (!resp.ok){ const t = await resp.text(); throw new Error(`HTTP ${resp.status}: ${t}`); }
        const data = await resp.json();
        const resolved = Array.isArray(data.input?.resolved) && data.input.resolved.length ? ` (matched: ${data.input.resolved.join(', ')})` : '';
        const infoText = `Input: ${data.input?.plant || v}${resolved}`;
        const goodArr = data.output?.good || [];
        const badArr = data.output?.avoid || [];
        renderNice(goodArr, badArr, infoText);
        // Also append a concise summary to chat
        const summary = [
          `Companion plan`,
          `Input: plant="${data.input?.plant || v}"${resolved}`,
          `Output:`,
          `  Helps: ${goodArr.length ? goodArr.join(', ') : '—'}`,
          `  Avoid: ${badArr.length ? badArr.join(', ') : '—'}`
        ].join('\n');
        appendChatBot(summary);
      } catch (err){
        box.innerHTML = '';
        box.appendChild(card('Error', String(err.message||err)));
      }
    });
  }

  function mockCompanions(p){
    const lower = String(p||'').toLowerCase();
    if (lower === 'tomato') return { good: ['basil','marigold','onion'], bad: ['fennel'] };
    if (lower === 'pepper') return { good: ['basil','onion'], bad: ['fennel'] };
    if (lower === 'cucumber') return { good: ['dill','radish','nasturtium'], bad: ['potato'] };
    if (lower === 'basil') return { good: ['tomato','pepper'], bad: [] };
    return { good: ['marigold'], bad: [] };
  }

  function mockCompanions(p){
    const lower = String(p||'').toLowerCase();
    if (lower === 'tomato') return { good: ['basil','marigold','onion'], bad: ['fennel'] };
    if (lower === 'pepper') return { good: ['basil','onion'], bad: ['fennel'] };
    if (lower === 'cucumber') return { good: ['dill','radish','nasturtium'], bad: ['potato'] };
    if (lower === 'basil') return { good: ['tomato','pepper'], bad: [] };
    return { good: ['marigold'], bad: [] };
  }

  // ---- Yield ----
  function initYield(){
    const crop = document.getElementById('ye-crop');
    const temp = document.getElementById('ye-temp');
    const rain = document.getElementById('ye-rain');
    const run = document.getElementById('ye-run');
    const out = document.getElementById('ye-result');
    run?.addEventListener('click', async () => {
      out.textContent = 'Calculating…';
      const payload = { crop: String(crop.value||'').trim(), temp: Number(temp.value||0), rain: Number(rain.value||0) };
      try {
        const resp = await fetch('/api/yield', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
        if (!resp.ok){ const t = await resp.text(); throw new Error(`HTTP ${resp.status}: ${t}`); }
        const data = await resp.json();
        const y = data.output?.yield_t_ha;
        const resolved = data.input?.resolved ? ` (matched: ${data.input.resolved})` : '';
        out.innerHTML = `
          <div class="cp-col">
            <div class="cp-header"><div class="cp-title">Yield estimate</div><div class="cp-count">t/ha</div></div>
            <div class="status">Input: crop=${data.input?.crop}${resolved}, temp=${data.input?.temp}, rain=${data.input?.rain}</div>
            <div class="tags" style="margin-top:8px"><span class="tag">Estimated: <b>${(y ?? 'n/a')}</b> t/ha</span></div>
          </div>`;
        const summary = [
          'Yield estimate',
          `Input: crop="${data.input?.crop}", temp=${data.input?.temp}, rain=${data.input?.rain}${resolved}`,
          `Output: estimated_yield=${y} t/ha`
        ].join('\n');
        appendChatBot(summary);
      } catch (err){
        out.innerHTML = '';
        out.appendChild(card('Error', String(err.message||err)));
      }
    });
  }

  // ---- Feedback ----
  function initFeedback(){
    const notes = document.getElementById('fb-notes');
    const save = document.getElementById('fb-save');
    const list = document.getElementById('fb-list');
    const status = document.getElementById('fb-status');
    let rating = 0;
    document.querySelectorAll('[data-fb]').forEach(btn => btn.addEventListener('click', () => {
      rating = btn.dataset.fb === 'up' ? 1 : -1;
      status.textContent = `Selected: ${rating === 1 ? '👍' : '👎'}`;
    }));
    save?.addEventListener('click', () => {
      const entry = { ts: Date.now(), rating, notes: notes.value.trim() };
      const arr = JSON.parse(localStorage.getItem('aig_feedback')||'[]');
      arr.unshift(entry); localStorage.setItem('aig_feedback', JSON.stringify(arr));
      notes.value=''; rating=0; status.textContent='Saved!';
      renderFBList(list, arr);
    });
    renderFBList(list, JSON.parse(localStorage.getItem('aig_feedback')||'[]'));
  }

  function renderFBList(ul, arr){
    ul.innerHTML = '';
    for (const e of arr){
      const li = document.createElement('li');
      const d = new Date(e.ts).toLocaleString();
      li.innerHTML = `<div class="sim">${d} — ${e.rating===1?'👍':'👎'}</div><div>${(e.notes||'').replace(/</g,'&lt;')}</div>`;
      ul.appendChild(li);
    }
  }

  // ---- Settings ----
  function initSettings(){
    // Reserved for future settings
  }

  // ---- helpers ----
  function card(title, body){
    const div = document.createElement('div');
    div.className = 'results';
    const li = document.createElement('li');
    li.innerHTML = `<div class="sim">${title}</div><div>${body.replace(/</g,'&lt;')}</div>`;
    div.appendChild(li);
    return div;
  }
  function tag(text, kind){
    const span = document.createElement('span');
    span.className = `tag ${kind||''}`;
    span.textContent = text;
    return span;
  }

  function chip(text, onClick){
    const span = document.createElement('span');
    span.className = 'chip';
    span.textContent = text;
    span.addEventListener('click', onClick);
    return span;
  }

  function initAppBar(){
    const btnNew = document.getElementById('btn-newchat');
    btnNew?.addEventListener('click', () => {
      newThread();
    });
  }

  // ---- Sidebar toggle ----
  function initSidebarToggle(){
    const btn = document.getElementById('btn-toggle-sidebar');
    const shell = document.querySelector('.chat-shell');
    const backdrop = document.getElementById('sidebar-backdrop');
    // restore saved state (default open on desktop)
    const saved = localStorage.getItem('aig_sidebar_open');
    const open = saved !== '0'; // Default to open unless explicitly closed
    setSidebarOpen(shell, open);
    btn?.addEventListener('click', () => {
      const isOpen = !shell.classList.contains('sidebar-closed');
      setSidebarOpen(shell, !isOpen);
    });
    // Close sidebar when clicking backdrop on mobile
    backdrop?.addEventListener('click', () => {
      setSidebarOpen(shell, false);
    });
  }
  function setSidebarOpen(shell, open){
    if (!shell) return;
    const backdrop = document.getElementById('sidebar-backdrop');
    if (open){ 
      shell.classList.remove('sidebar-closed');
      shell.classList.add('sidebar-open'); 
      // On mobile, show backdrop
      if (window.innerWidth <= 1000 && backdrop) {
        backdrop.classList.remove('hidden');
      }
      localStorage.setItem('aig_sidebar_open','1'); 
    }
    else { 
      shell.classList.add('sidebar-closed'); 
      shell.classList.remove('sidebar-open');
      if (backdrop) backdrop.classList.add('hidden');
      localStorage.setItem('aig_sidebar_open','0'); 
    }
  }

  // Append a bot message to the chat area and persist it
  function appendChatBot(text){
    try {
      const messagesEl = document.getElementById('ch-messages');
      const history = loadChat();
      history.push({ role: 'assistant', content: String(text||'').trim(), ts: Date.now() });
      saveChat(history);
      renderChat(messagesEl, history, true);
    } catch (e) { /* no-op */ }
  }

  // ---- Chat ----
  function initChat(){
    const messagesEl = document.getElementById('ch-messages');
    const input = document.getElementById('ch-input');
    const send = document.getElementById('ch-send');
    const clear = document.getElementById('ch-clear');
    let sending = false;

    renderChat(messagesEl, loadChat());
    renderSuggestions();

    function doSend(){
      if (sending) return;
      const text = (input.value || '').trim();
      if (!text) return;
      input.value = '';
  
  // Store image data URL if present
  let imageDataUrl = null;
  if (window.chatImageFile) {
    const reader = new FileReader();
    reader.onload = (e) => {
      imageDataUrl = e.target.result;
      // Now proceed with sending
      proceedWithSend(text, imageDataUrl);
    };
    reader.readAsDataURL(window.chatImageFile);
  } else {
    proceedWithSend(text, null);
  }
}

function proceedWithSend(text, imageDataUrl) {
  let history = loadChat();
  const userMsg = { role:'user', content:text, ts: Date.now() };
  if (imageDataUrl) userMsg.image = imageDataUrl;
  history.push(userMsg);
  saveChat(history); renderChat(messagesEl, history, true);
      hideSuggestions();

      sending = true; send.disabled = true;
      // Live mode: call agent /api/chat
      showTyping(messagesEl);
      (async () => {
        try {
          const fd = new FormData();
          fd.append('thread_id', currentThreadId());
          fd.append('message', text);
          
          // Attach image if present
          if (window.chatImageFile) {
            fd.append('image', window.chatImageFile);
            // Clear the image after sending
            window.chatImageFile = null;
            const chatAttach = document.getElementById('ch-attach-preview');
            if (chatAttach) chatAttach.innerHTML = '';
          }
          
          const resp = await fetch('/api/chat', { method: 'POST', body: fd });
          if (!resp.ok){ const t = await resp.text(); throw new Error(`HTTP ${resp.status}: ${t}`); }
          const data = await resp.json();
          const reply = String(data.message || '').trim() || '[no response]';
          let hist2 = loadChat();
          hist2.push({ role:'assistant', content: reply, ts: Date.now() });
          saveChat(hist2); renderChat(messagesEl, hist2, true);
        } catch (err){
          let hist2 = loadChat();
          hist2.push({ role:'assistant', content: 'Error: ' + (err.message||String(err)), ts: Date.now() });
          saveChat(hist2); renderChat(messagesEl, hist2, true);
        } finally {
          sending = false; send.disabled = false; hideTyping(messagesEl);
        }
      })();
    }

    send?.addEventListener('click', doSend);
    input?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); doSend(); }
    });
    clear?.addEventListener('click', () => { saveChat([]); renderChat(messagesEl, []); renderSuggestions(); });
  }

  function renderChat(el, history, stick=false){
    el.innerHTML = '';
    for (const m of history){
      const row = document.createElement('div');
      row.className = `msg ${m.role === 'user' ? 'user' : 'bot'}`;
      const avatar = document.createElement('div');
      avatar.className = 'avatar';
      avatar.textContent = m.role === 'user' ? 'You' : 'AIG';
      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      
      // Add image if present
      if (m.image) {
        const img = document.createElement('img');
        img.src = m.image;
        img.style.maxWidth = '200px';
        img.style.borderRadius = '8px';
        img.style.marginBottom = '8px';
        img.style.display = 'block';
        bubble.appendChild(img);
      }
      
      const textNode = document.createElement('div');
      textNode.innerHTML = formatMessageHTML(String(m.content||''));
      bubble.appendChild(textNode);
      
      const ts = document.createElement('div');
      ts.className = 'ts';
      ts.textContent = new Date(m.ts).toLocaleTimeString();
      if (m.role === 'user'){ row.appendChild(ts); row.appendChild(bubble); row.appendChild(avatar); }
      else { row.appendChild(avatar); row.appendChild(bubble); row.appendChild(ts); }
      el.appendChild(row);
    }
    if (stick) el.scrollTop = el.scrollHeight;
  }

  // Safely format message text: escape HTML, linkify URLs, preserve newlines, process markdown
  function formatMessageHTML(str){
    try {
      const escaped = String(str)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;');
      
      // Process markdown formatting
      const withMarkdown = escaped
        // Bold text: **text** -> <strong>text</strong>
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        // Italic text: *text* -> <em>text</em> (single asterisk, not preceded/followed by asterisk)
        .replace(/(?<!\*)\*([^*\s][^*]*[^*\s]|\S)\*(?!\*)/g, '<em>$1</em>')
        // Code: `text` -> <code>text</code>
        .replace(/`([^`]+?)`/g, '<code>$1</code>');
      
      const linked = withMarkdown.replace(/(https?:\/\/[^\s]+)|(www\.[^\s]+)/g, (m) => {
        const url = m.startsWith('http') ? m : `https://${m}`;
        return `<a href="${url}" target="_blank" rel="noopener noreferrer">${m}</a>`;
      });
      return linked.replace(/\n/g, '<br/>');
    } catch { return String(str||''); }
  }

  function renderSuggestions(){
    const sug = document.getElementById('ch-suggestions');
    if (!sug) return;
    const presets = [
      'What grows well with tomatoes?',
      'Diagnose yellow spots on my pepper leaves',
      'Estimate yield for cucumbers at 24°C and 80mm rain',
      'Plan companions for basil and marigold'
    ];
    sug.innerHTML = '';
    presets.forEach(p => sug.appendChild(chip(p, () => {
      const input = document.getElementById('ch-input');
      input.value = p; input.focus();
    })));
  }
  function hideSuggestions(){ const sug = document.getElementById('ch-suggestions'); if (sug) sug.innerHTML = ''; }

  function showTyping(el){
    const row = document.createElement('div'); row.className = 'msg bot'; row.id = 'typing-row';
    const avatar = document.createElement('div'); avatar.className = 'avatar'; avatar.textContent = 'AIG';
    const bubble = document.createElement('div'); bubble.className = 'bubble';
    bubble.innerHTML = '<span class="typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span>';
    const ts = document.createElement('div'); ts.className = 'ts'; ts.textContent = new Date().toLocaleTimeString();
    row.appendChild(avatar); row.appendChild(bubble); row.appendChild(ts);
    el.appendChild(row); el.scrollTop = el.scrollHeight;
  }
  function hideTyping(el){ const r = document.getElementById('typing-row'); if (r) r.remove(); }

  // ---- Threads (multi-chat) ----
  function genId(){ return Math.random().toString(36).slice(2, 10); }
  function loadThreads(){ try { return JSON.parse(localStorage.getItem('aig_threads')||'[]'); } catch { return []; } }
  function saveThreads(arr){ localStorage.setItem('aig_threads', JSON.stringify(arr)); }
  function currentThreadId(){ return localStorage.getItem('aig_current_thread')||''; }
  function setCurrentThreadId(id){ localStorage.setItem('aig_current_thread', id); }
  function migrateSingleChat(){
    const single = (()=>{ try { return JSON.parse(localStorage.getItem('aig_chat')||'[]'); } catch { return []; } })();
    if (!single || single.length===0) return null;
    const id = genId();
    const title = deriveTitle(single) || 'Conversation';
    const ts = single[0]?.ts || Date.now();
    const t = { id, title, ts, messages: single };
    const threads = loadThreads(); threads.unshift(t); saveThreads(threads);
    setCurrentThreadId(id);
    return t;
  }
  function ensureThread(){
    let threads = loadThreads();
    if (threads.length===0){
      migrateSingleChat();
      threads = loadThreads();
      if (threads.length===0){
        const id = genId();
        const t = { id, title:'New chat', ts: Date.now(), messages: [] };
        threads = [t]; saveThreads(threads); setCurrentThreadId(id);
      }
    }
    if (!currentThreadId()) setCurrentThreadId(threads[0].id);
  }
  function deriveTitle(messages){
    const firstUser = (messages||[]).find(m => m.role==='user');
    const raw = firstUser?.content?.trim() || '';
    return raw ? (raw.length>40 ? raw.slice(0,37)+'…' : raw) : 'Conversation';
  }
  function threadById(id){ return loadThreads().find(t => t.id===id); }
  function getCurrentThread(){ return threadById(currentThreadId()); }
  function setThreadMessages(id, messages){
    const threads = loadThreads();
    const idx = threads.findIndex(t => t.id===id);
    if (idx===-1) return;
    threads[idx].messages = messages.slice(-200);
    threads[idx].title = deriveTitle(messages) || threads[idx].title;
    threads[idx].ts = messages[0]?.ts || threads[idx].ts;
    saveThreads(threads);
  }
  function newThread(){
    const t = { id: genId(), title: 'New chat', ts: Date.now(), messages: [] };
    const threads = loadThreads(); threads.unshift(t); saveThreads(threads); setCurrentThreadId(t.id);
    renderThreadList();
    const messagesEl = document.getElementById('ch-messages');
    renderChat(messagesEl, []);
    renderSuggestions();
  }
  function removeThread(id){
    let threads = loadThreads();
    const idx = threads.findIndex(t => t.id===id);
    if (idx<0) return;
    threads.splice(idx,1); saveThreads(threads);
    if (currentThreadId()===id){ setCurrentThreadId(threads[0]?.id || ''); }
    renderThreadList();
    const cur = getCurrentThread();
    const messagesEl = document.getElementById('ch-messages');
    renderChat(messagesEl, cur?.messages||[]);
  }
  function initChatThreads(){ ensureThread(); renderThreadList(); document.getElementById('sb-new')?.addEventListener('click', newThread); }
  function renderThreadList(){
    const ul = document.getElementById('sb-threads'); if (!ul) return;
    const curId = currentThreadId(); const threads = loadThreads();
    ul.innerHTML='';
    for (const t of threads){
      const li = document.createElement('li'); li.className = 'sb-item'+(t.id===curId?' active':'');
      const name = document.createElement('div'); name.className='sb-name'; name.textContent = t.title || 'Conversation';
      const meta = document.createElement('div'); meta.className='sb-meta';
      const when = new Date(t.ts||Date.now()).toLocaleDateString();
      const count = (t.messages||[]).length + ' msgs';
      meta.innerHTML = `<span>${when}</span><span>${count}</span>`;
      const actions = document.createElement('div'); actions.className='sb-actions';
      const del = document.createElement('button'); del.className='btn-ghost'; del.textContent='Delete'; del.addEventListener('click', (e)=>{ e.stopPropagation(); removeThread(t.id); });
      actions.appendChild(del);
      li.appendChild(name); li.appendChild(meta); li.appendChild(actions);
      li.addEventListener('click', () => { setCurrentThreadId(t.id); renderThreadList(); const mEl = document.getElementById('ch-messages'); renderChat(mEl, t.messages||[], true); });
      ul.appendChild(li);
    }
  }
  function loadChat(){ const t = getCurrentThread(); return t?.messages || []; }
  function saveChat(arr){ const id = currentThreadId(); if (id) setThreadMessages(id, arr); }

  function init(){
    bindTabs();
      setActiveTab('chat');
    initSearch();
    initAnalyze();
    initCompanions();
    initYield();
    initFeedback();
    initSettings();
    initSidebarToggle();
    initChatThreads();
    initChat();
    initAppBar();
      initChatTools();
      initChatCamera();
      initPageClose();
  }

  // ---- Page close buttons ----
  function initPageClose(){
    document.querySelectorAll('.page-close').forEach(btn => {
      btn.addEventListener('click', () => setActiveTab('chat'));
    });
  }

  // ---- Chat tools dropdown ----
  function initChatTools(){
    const btn = document.getElementById('ch-tools');
    const menu = document.getElementById('tools-menu');
    const close = document.getElementById('tools-close');
    const items = menu?.querySelectorAll('[data-go]') || [];

    const open = () => menu?.classList.add('open');
    const hide = () => menu?.classList.remove('open');
    btn?.addEventListener('click', (e) => { e.stopPropagation(); if (menu?.classList.contains('open')) hide(); else open(); });
    close?.addEventListener('click', hide);
    items.forEach(el => el.addEventListener('click', () => {
      const dest = el.getAttribute('data-go');
      const label = (el.textContent || '').toLowerCase();
      if (dest === 'analyze'){
        const low = label.toLowerCase();
        if (low.includes('image') || low.includes('crop')) analyzeMode = 'detect';
        else if (low.includes('disease')) analyzeMode = 'disease';
        updateAnalyzeUI();
      }
      if (dest) setActiveTab(dest);
      hide();
    }));
    document.addEventListener('click', (e) => {
      if (!menu) return;
      if (!menu.contains(e.target) && e.target !== btn){ hide(); }
    });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') hide(); });
  }

  // Toggle Analyze UI for crop vs disease modes
  function updateAnalyzeUI(){
    try {
      const btn = document.getElementById('an-run');
      if (!btn) return;
      if (analyzeMode === 'detect'){
        btn.textContent = 'Detect crop';
      } else {
        btn.textContent = 'Diagnose disease';
      }
    } catch (e) { /* no-op */ }
  }

  // ---- Chat camera functionality ----
  let cameraStream = null;
  let currentFacingMode = 'environment'; // Start with back camera
  
  function initChatCamera(){
    const btn = document.getElementById('ch-camera');
    const menu = document.getElementById('camera-menu');
    const close = document.getElementById('camera-close');
    const takeBtn = document.getElementById('cam-take');
    const uploadBtn = document.getElementById('cam-upload-btn');
    const inputCapture = document.getElementById('cam-capture');
    const inputUpload = document.getElementById('cam-upload');

    // Camera modal elements
    const modal = document.getElementById('camera-modal');
    const modalClose = document.getElementById('camera-modal-close');
    const video = document.getElementById('camera-video');
    const canvas = document.getElementById('camera-canvas');
    const captureBtn = document.getElementById('camera-capture-btn');
    const switchBtn = document.getElementById('camera-switch-btn');
    const preview = document.getElementById('camera-preview');
    const previewImg = document.getElementById('camera-preview-img');
    const retakeBtn = document.getElementById('camera-retake-btn');
    const useBtn = document.getElementById('camera-use-btn');

    const openMenu = () => menu?.classList.add('open');
    const hideMenu = () => menu?.classList.remove('open');
    
    btn?.addEventListener('click', (e) => { 
      e.stopPropagation(); 
      if (menu?.classList.contains('open')) hideMenu(); 
      else openMenu(); 
    });
    close?.addEventListener('click', hideMenu);
    document.addEventListener('click', (e) => { 
      if (!menu) return; 
      if (!menu.contains(e.target) && e.target !== btn){ hideMenu(); } 
    });
    document.addEventListener('keydown', (e) => { 
      if (e.key === 'Escape') { hideMenu(); closeCamera(); }
    });

    // Real camera capture
    takeBtn?.addEventListener('click', async () => { 
      hideMenu(); 
      
      // Check if camera is supported
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('Camera is not supported in this browser. Please use file upload instead.');
        inputUpload?.click();
        return;
      }
      
      await openCamera(); 
    });
    
    // File upload fallback
    uploadBtn?.addEventListener('click', () => { 
      hideMenu(); 
      inputUpload?.click(); 
    });

    // Modal controls
    modalClose?.addEventListener('click', closeCamera);
    captureBtn?.addEventListener('click', capturePhoto);
    switchBtn?.addEventListener('click', switchCamera);
    retakeBtn?.addEventListener('click', retakePhoto);
    useBtn?.addEventListener('click', usePhoto);

    // File input handlers
    inputCapture?.addEventListener('change', () => {
      const f = inputCapture.files?.[0]; 
      if (f) handlePickedImage(f);
      inputCapture.value = '';
    });
    
    inputUpload?.addEventListener('change', () => {
      const f = inputUpload.files?.[0]; 
      if (f) handlePickedImage(f);
      inputUpload.value = '';
    });

    // Camera functions
    async function openCamera() {
      try {
        if (modal) modal.classList.remove('hidden');
        if (preview) preview.classList.add('hidden');
        
        // Show loading state
        if (captureBtn) {
          captureBtn.disabled = true;
          captureBtn.textContent = '📷 Loading camera...';
        }
        
        const constraints = {
          video: {
            facingMode: currentFacingMode,
            width: { ideal: 1280 },
            height: { ideal: 720 }
          }
        };

        cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
        if (video) {
          video.srcObject = cameraStream;
          
          // Wait for video to load
          video.addEventListener('loadedmetadata', () => {
            if (captureBtn) {
              captureBtn.disabled = false;
              captureBtn.textContent = '📸 Capture Photo';
            }
          });
          
          video.play();
        }
      } catch (error) {
        console.error('Error accessing camera:', error);
        let errorMessage = 'Could not access camera. ';
        
        if (error.name === 'NotAllowedError') {
          errorMessage += 'Please allow camera access in your browser settings.';
        } else if (error.name === 'NotFoundError') {
          errorMessage += 'No camera found on this device.';
        } else {
          errorMessage += 'Please check permissions or use file upload instead.';
        }
        
        alert(errorMessage);
        closeCamera();
      }
    }

    function closeCamera() {
      if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
      }
      if (modal) modal.classList.add('hidden');
      if (preview) preview.classList.add('hidden');
    }

    function capturePhoto() {
      if (!video || !canvas) return;
      
      // Add visual feedback - flash effect
      if (video) {
        video.style.filter = 'brightness(1.5)';
        setTimeout(() => {
          video.style.filter = 'brightness(1)';
        }, 150);
      }
      
      const context = canvas.getContext('2d');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      const imageDataUrl = canvas.toDataURL('image/jpeg', 0.8);
      if (previewImg) {
        previewImg.src = imageDataUrl;
        preview?.classList.remove('hidden');
      }
    }

    async function switchCamera() {
      currentFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';
      closeCamera();
      await openCamera();
    }

    function retakePhoto() {
      if (preview) preview.classList.add('hidden');
    }

    function usePhoto() {
      if (!previewImg) return;
      
      // Convert data URL to blob
      canvas.toBlob((blob) => {
        if (blob) {
          // Create a file from the blob
          const file = new File([blob], `camera-capture-${Date.now()}.jpg`, {
            type: 'image/jpeg'
          });
          handlePickedImage(file);
          closeCamera();
        }
      }, 'image/jpeg', 0.8);
    }
  }

  function handlePickedImage(file){
    // Image picked from chat - attach to chat input instead of switching to Analyze page
    const chatAttach = document.getElementById('ch-attach-preview');
    const chatSend = document.getElementById('ch-send');
    
    if (chatAttach && chatSend) {
      // Store the file for chat to use
      window.chatImageFile = file;
      
      // Show preview in chat
      chatAttach.innerHTML = '';
      const wrapper = document.createElement('div');
      wrapper.className = 'attach-item';
      const img = document.createElement('img');
      img.src = URL.createObjectURL(file);
      img.style.maxHeight = '60px';
      img.style.borderRadius = '4px';
      const removeBtn = document.createElement('button');
      removeBtn.textContent = '×';
      removeBtn.style.cssText = 'margin-left:8px;padding:2px 8px;cursor:pointer;';
      removeBtn.onclick = () => {
        window.chatImageFile = null;
        chatAttach.innerHTML = '';
      };
      wrapper.appendChild(img);
      wrapper.appendChild(removeBtn);
      chatAttach.appendChild(wrapper);
      
      // Focus on chat input
      const chatInput = document.getElementById('ch-input');
      if (chatInput) chatInput.focus();
    }
  }

  return { init };
})();
