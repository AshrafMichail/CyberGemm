(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const r of document.querySelectorAll('link[rel="modulepreload"]'))o(r);new MutationObserver(r=>{for(const a of r)if(a.type==="childList")for(const l of a.addedNodes)l.tagName==="LINK"&&l.rel==="modulepreload"&&o(l)}).observe(document,{childList:!0,subtree:!0});function n(r){const a={};return r.integrity&&(a.integrity=r.integrity),r.referrerPolicy&&(a.referrerPolicy=r.referrerPolicy),r.crossOrigin==="use-credentials"?a.credentials="include":r.crossOrigin==="anonymous"?a.credentials="omit":a.credentials="same-origin",a}function o(r){if(r.ep)return;r.ep=!0;const a=n(r);fetch(r.href,a)}})();const K=`# CyberGemm starter kernel
# This is the intentionally boring baseline.
#
# It is meant to be the simplest possible correct program, not a good one:
# - only compute core 0 does useful work
# - every other compute core exits immediately
# - core 0 computes every output tile one at a time
# - each global load waits before issuing the next instruction
# - each tensor op waits before issuing the next instruction
# - the shared-memory layout is the naive layout, so bank conflicts remain
#
# That makes it easy to understand and easy to beat. Good kernels should split
# output tiles across cores, overlap gld.async with mma.async, use pad1 shared
# layouts, and keep many tensor op cores busy at once.

# r0 starts as the hardware core id. Machine constants such as coreid,
# total_tiles, k_tiles, and num_cores are read-only pseudo-registers provided by
# the simulator.
mov r0, coreid

# The default kernel is single-worker by design. If this is not core 0, jump to
# done and halt. This is correct because core 0 will eventually write every C
# tile, but it wastes almost the whole machine.
jnz r0, done

# Reuse r0 as the linear output tile index. Core 0 starts with tile 0 and later
# increments by 1, so it visits every output tile serially.
mov r0, 0

tile_loop:
# Stop once all output tiles have been written. With the GB300-ish default
# problem this means core 0 loops over 640 C tiles by itself.
jge r0, total_tiles, done

# Convert linear output tile r0 into two tile coordinates:
#   r1 = M/output-row tile
#   r2 = N/output-column tile
# The simulator handles edge tiles when the matrix dimensions are not exact
# multiples of 16.
tile.coords r1, r2, r0

# acc0 holds the 16x16 C tile being accumulated across K. Clear it before the
# K loop starts, otherwise values from the previous output tile would leak in.
clear acc0

# r3 is the K-tile loop counter. Each iteration multiplies one A tile and one B
# tile, then accumulates that product into acc0.
mov r3, 0

k_loop:
# Each output tile needs k_tiles tensor ops. For the default problem k_tiles is
# 16, so this tiny loop runs 16 times for each C tile.
jge r3, k_tiles, store_tile

# Load A[r1, r3] from global memory into shared tile slot t0.
# This uses the default naive shared-memory layout. It is simple, but its stride
# creates bank conflicts in the tensor op. Adding ", pad1" would avoid those.
gld.async t0, A, r1, r3

# This wait makes the program easy to reason about and very slow: no B load or
# tensor work is allowed to overlap the A load.
wait all

# Load B[r3, r2] into shared tile slot t1. This is also naive layout and also
# waits immediately.
gld.async t1, B, r3, r2

# Wait until B is present in shared memory. A faster program would have already
# prefetched this tile while previous tensor work was running.
wait all
mma.async acc0, t0, t1

# Wait for the tensor op to finish before issuing the next K iteration. This
# serializes all 16 tensor ops for the tile and leaves most tensor units idle.
wait all

# Move to the next K tile and repeat the load/load/mma sequence.
add r3, r3, 1
jmp k_loop

store_tile:
# Store the completed accumulator tile to C[r1, r2]. Edge rows/columns are
# clipped automatically by the simulator when this is a partial tile.
gst.async C, r1, r2, acc0

# Wait for the store before starting the next C tile. This is another deliberate
# serialization point that optimized kernels should avoid when possible.
wait all

# Advance to the next linear output tile. Since only core 0 is active, this is
# +1 instead of +num_cores.
add r0, r0, 1
jmp tile_loop

done:
# All nonzero cores get here immediately; core 0 gets here after writing every
# output tile. halt ends this core's program.
halt
`,q=[{syntax:"mov dst, src",description:"Copy an integer constant, register, or machine constant into a register."},{syntax:"add/sub/mul/div/mod/min dst, a, b",description:"Integer arithmetic for loop counters, tile assignment, and address math."},{syntax:"jmp label, jz reg, label, jnz reg, label, jlt a, b, label, jge a, b, label",description:"Control flow. Labels use the form name: on their own line or before an instruction."},{syntax:"tile.coords rowTile, colTile, tileIndex",description:"Convert a linear output tile index into M/N tile coordinates for the fixed GEMM grid."},{syntax:"gld.async tile, A|B, rowTile, colTile[, pad1]",description:"Asynchronously copy one global A or B tile into a fixed shared-memory tile slot. The optional pad1 layout changes the shared stride to reduce bank conflicts."},{syntax:"mma.async acc, tileA, tileB",description:"Asynchronously enqueue a 16x16x16 tensor operation on one of the shared tensor op cores."},{syntax:"gst.async C, rowTile, colTile, acc",description:"Asynchronously store an accumulator tile to global C with edge bounds checks."},{syntax:"clear acc",description:"Reset an accumulator tile before starting a new output tile."},{syntax:"wait all",description:"Stall the issuing compute core until its outstanding async memory, tensor, and store operations complete."},{syntax:"barrier",description:"Synchronize all active compute cores at the current program point."},{syntax:"halt",description:"Stop the current compute core. Halting with pending async work is reported as an error."}],d={computeCores:640,tensorCores:640,frequencyGHz:2.86,cacheLineBytes:128,sharedMemoryBanks:32,bankWidthBytes:4,sharedBytesPerCore:228*1024,registerCount:64,globalBaseLatency:4,globalBytesPerCycle:1e6,globalTransactionPenalty:0,globalContentionPenalty:0,tensorLatency:8,tensorBankConflictPenalty:8,storeBaseLatency:4,barrierLatency:8,issueLatency:1,maxInstructions:2e6,tileM:16,tileN:16,tileK:16,matrixM:512,matrixN:320,matrixK:256};function C(e=d){return{mTiles:Math.ceil(e.matrixM/e.tileM),nTiles:Math.ceil(e.matrixN/e.tileN),kTiles:Math.ceil(e.matrixK/e.tileK),totalTiles:Math.ceil(e.matrixM/e.tileM)*Math.ceil(e.matrixN/e.tileN)}}function V(e=d){return 2*e.matrixM*e.matrixN*e.matrixK}function Z(e=d){return 2*e.tileM*e.tileN*e.tileK}function U(e=d){const t=C(e),n=t.totalTiles*t.kTiles;return Math.ceil(n/e.tensorCores)*e.tensorLatency}function J(e=d){const t=U(e)/(e.frequencyGHz*1e9);return V(e)/t/1e12}function Q(e=d){const t=C(e),n=t.totalTiles*t.kTiles*Z(e),o=U(e)/(e.frequencyGHz*1e9);return n/o/1e12}const y=[{id:"split-work",title:"1. Single-Core Lockdown",bottleneck:"Dormant cores detected: one core owns the trace while the rest sit dark.",objective:"Break the lockdown so every output tile is claimed across the compute grid.",hint:"Initialize r0 from coreid, then stride through tile_loop by num_cores instead of stepping by 1.",maxTflops:50.829},{id:"avoid-banks",title:"2. Bank Collision Gridlock",bottleneck:"Shared-memory bank collisions are storming the local grid.",objective:"Route the shared-memory layout with zero bank-conflict penalty cycles.",hint:"Enable the pad1 layout on both A and B gld.async instructions.",maxTflops:599.785},{id:"batch-loads",title:"3. Memory Pipeline Jam",bottleneck:"HBM traffic jam: A and B transfers are serialized by lonely waits.",objective:"Launch both A and B transfers before waiting so wait all sees a packet burst.",hint:"Move wait all after both gld.async instructions, then let the packet burst land together.",maxTflops:749.732},{id:"overlap-tensor",title:"4. Tensor Overlap Breach",bottleneck:"Tensor array underfed: memory and MMA still run in separate trace phases.",objective:"Breach the pipeline: at least one wait must cover memory transfer and MMA together.",hint:"Double-buffer shared tiles: fetch the next K tile while the current mma.async is already in flight.",maxTflops:1230.329}];function $(e,t,n){const o=n.filter(r=>r.severity==="error");if(o.length>0)return{passed:!1,status:"Trace rejected. Fix runtime or assembly errors first.",details:o.slice(0,3).map(r=>r.message)};if(!t.outputMatchesReference)return{passed:!1,status:"C buffer compromised. Correctness must pass before this mission unlocks.",details:[`Corruption delta: max error ${t.maxError.toExponential(3)}.`]};switch(e.id){case"split-work":return ne(t);case"avoid-banks":return oe(t);case"batch-loads":return re(t);case"overlap-tensor":return ae(t)}}function X(e){return e.coreProfiles.filter(t=>t.tilesCompleted>0).length}function Y(e){return e.coreProfiles.reduce((t,n)=>t+n.bankConflictCycles,0)}function ee(e){return e.coreProfiles.some(t=>t.timeline.some(n=>n.waitingOn.filter(o=>o.kind==="global").length>=2))}function te(e){return e.coreProfiles.some(t=>t.timeline.some(n=>{const o=new Set(n.waitingOn.map(r=>r.kind));return o.has("global")&&o.has("tensor")}))}function ne(e){const t=Math.min(d.computeCores,C(d).totalTiles),n=X(e),o=n>=t;return{passed:o,status:o?"Stall signature eliminated: the compute grid is online.":"Single-core lockdown remains: too few cores are carrying tiles.",details:[`Active tile lanes: ${n} / ${t}.`]}}function oe(e){const t=Y(e),n=t===0;return{passed:n,status:n?"Stall signature eliminated: bank collisions are gone.":"Bank collision storm still disrupts shared memory.",details:[`Collision tax: ${t.toLocaleString()} bank-conflict cycles.`]}}function re(e){const t=ee(e);return{passed:t,status:t?"Stall signature eliminated: A/B transfers now launch as a burst.":"HBM traffic jam remains: loads still arrive one at a time.",details:[t?"Found a wait all with at least two global-memory transfers pending.":"Issue A and B gld.async first, then wait all."]}}function ae(e){const t=te(e);return{passed:t,status:t?"Stall signature eliminated: tensor work overlaps the memory trace.":"Tensor array underfed: memory and MMA still take turns.",details:[t?"Found a wait all that was simultaneously waiting on global memory and tensor work.":"Keep a next tile load in flight while mma.async works on the current tile."]}}const se=`# Temporary level 1 solution: distribute output tiles.
mov r0, coreid

tile_loop:
jge r0, total_tiles, done
tile.coords r1, r2, r0
clear acc0
mov r3, 0

k_loop:
jge r3, k_tiles, store_tile
gld.async t0, A, r1, r3
wait all
gld.async t1, B, r3, r2
wait all
mma.async acc0, t0, t1
wait all
add r3, r3, 1
jmp k_loop

store_tile:
gst.async C, r1, r2, acc0
wait all
add r0, r0, num_cores
jmp tile_loop

done:
halt
`,le=`# Temporary level 2 solution: distribute work and use pad1 shared layout.
mov r0, coreid

tile_loop:
jge r0, total_tiles, done
tile.coords r1, r2, r0
clear acc0
mov r3, 0

k_loop:
jge r3, k_tiles, store_tile
gld.async t0, A, r1, r3, pad1
wait all
gld.async t1, B, r3, r2, pad1
wait all
mma.async acc0, t0, t1
wait all
add r3, r3, 1
jmp k_loop

store_tile:
gst.async C, r1, r2, acc0
wait all
add r0, r0, num_cores
jmp tile_loop

done:
halt
`,ie=`# Temporary level 3 solution: batch A/B loads before waiting.
mov r0, coreid

tile_loop:
jge r0, total_tiles, done
tile.coords r1, r2, r0
clear acc0
mov r3, 0

k_loop:
jge r3, k_tiles, store_tile
gld.async t0, A, r1, r3, pad1
gld.async t1, B, r3, r2, pad1
wait all
mma.async acc0, t0, t1
wait all
add r3, r3, 1
jmp k_loop

store_tile:
gst.async C, r1, r2, acc0
wait all
add r0, r0, num_cores
jmp tile_loop

done:
halt
`;function ce(){const{kTiles:e}=C(d),t=["# Temporary level 4 solution: double-buffer loads with tensor work.","mov r0, coreid","","tile_loop:","jge r0, total_tiles, done","tile.coords r1, r2, r0","clear acc0","","gld.async a0, A, r1, 0, pad1","gld.async b0, B, 0, r2, pad1","wait all",""];for(let o=0;o<e-1;o+=1){const r=o%2,a=(o+1)%2;t.push(`gld.async a${a}, A, r1, ${o+1}, pad1`,`gld.async b${a}, B, ${o+1}, r2, pad1`,`mma.async acc0, a${r}, b${r}`,"wait all","")}const n=(e-1)%2;return t.push(`mma.async acc0, a${n}, b${n}`,"wait all","gst.async C, r1, r2, acc0","wait all","add r0, r0, num_cores","jmp tile_loop","","done:","halt",""),t.join(`
`)}const de={"split-work":se,"avoid-banks":le,"batch-loads":ie,"overlap-tensor":ce()};function ue(e){return de[e]}const z={mov:{min:2,max:2},add:{min:3,max:3},sub:{min:3,max:3},mul:{min:3,max:3},div:{min:3,max:3},mod:{min:3,max:3},min:{min:3,max:3},jmp:{min:1,max:1,labelArg:0},jz:{min:2,max:2,labelArg:1},jnz:{min:2,max:2,labelArg:1},jlt:{min:3,max:3,labelArg:2},jge:{min:3,max:3,labelArg:2},"tile.coords":{min:3,max:3},"gld.async":{min:4,max:5},"gst.async":{min:4,max:4},"mma.async":{min:3,max:3},clear:{min:1,max:1},wait:{min:1,max:1},barrier:{min:0,max:0},halt:{min:0,max:0}};function M(e){const t=new Map,n=[],o=[];e.split(/\r?\n/).forEach((r,a)=>{const l=a+1;let s=me(r).trim();for(;s.length>0;){const f=/^([A-Za-z_][\w.]*):/.exec(s);if(!f)break;const w=f[1];t.has(w)?o.push({line:l,severity:"error",message:`Duplicate label "${w}".`}):t.set(w,n.length),s=s.slice(f[0].length).trim()}if(s.length===0)return;const i=/^([A-Za-z_.][\w.]*)\s*(.*)$/.exec(s);if(!i){o.push({line:l,severity:"error",message:"Expected an instruction."});return}const c=i[1].toLowerCase(),m=pe(i[2]),u=z[c];if(!u)o.push({line:l,severity:"error",message:`Unknown instruction "${c}".`});else if(m.length<u.min||m.length>u.max){const f=u.min===u.max?`${u.min}`:`${u.min}-${u.max}`;o.push({line:l,severity:"error",message:`"${c}" expects ${f} argument(s), got ${m.length}.`})}n.push({op:c,args:m,line:l,raw:r})});for(const r of n){const a=z[r.op];if(a?.labelArg===void 0||r.args.length<=a.labelArg)continue;const l=r.args[a.labelArg];t.has(l)||o.push({line:r.line,severity:"error",message:`Unknown label "${l}".`})}return{instructions:n,labels:t,diagnostics:o}}function me(e){const t=e.indexOf("#"),n=e.indexOf("//"),o=e.indexOf(";"),r=[t,n,o].filter(l=>l>=0),a=r.length>0?Math.min(...r):-1;return a>=0?e.slice(0,a):e}function pe(e){const t=e.trim();return t.length===0?[]:t.includes(",")?t.split(",").map(n=>n.trim()).filter(Boolean):t.split(/\s+/).filter(Boolean)}const I=5e3;let b=null;function he(e){const t={source:K,mode:"edit",result:null,currentLevelIndex:0,unlockedLevelIndex:0,isRunning:!1,runStartedAt:null,activeRunId:0},n=document.createElement("div");n.className="app";const o=document.createElement("aside");o.className="left-pane";const r=document.createElement("main");r.className="right-pane",n.append(o,r),e.replaceChildren(n);const a=()=>{fe(o,t,a),be(r,t,a)};a()}function fe(e,t,n){const o=C(d);e.replaceChildren(h("CyberGemm",[v("A neon GPU ops puzzle where one GEMM kernel is trapped inside a hostile trace."),ge(t)]),h("Mission Deck",[Ie(t,n)]),h("GPU Rig",[Fe([["Frequency",`${d.frequencyGHz} GHz`],["Compute",String(d.computeCores)],["Tensor",String(d.tensorCores)],["Cache line",`${d.cacheLineBytes} B`],["Banks",String(d.sharedMemoryBanks)],["Bank width",`${d.bankWidthBytes} B`],["Shared/core",`${d.sharedBytesPerCore/1024} KiB`],["Registers",String(d.registerCount)],["GEMM peak",`${J(d).toFixed(0)} TFLOPS`],["Tensor peak",`${Q(d).toFixed(0)} TFLOPS`]])]),h("Matrix Target",[L([["GEMM",`${d.matrixM} x ${d.matrixN} x ${d.matrixK}`],["Tensor tile","16 x 16 x 16"],["Output tiles",`${o.totalTiles} (${o.mTiles} x ${o.nTiles})`],["K tiles",String(o.kTiles)]]),v(`To breach peak throughput, overlap loads with tensor pulses, route around bank collisions, and spread ${o.totalTiles} tiles across ${d.computeCores} cores.`)]),h("Opcode Deck",[ze()]))}function ge(e){const t=document.createElement("div");t.className="operator-panel";const n=document.createElement("div");n.className="operator-portrait",n.textContent="CG";const o=document.createElement("div"),r=document.createElement("strong");r.textContent="Kernel daemon";const a=v(ye(e));return o.append(r,a),t.append(n,o),t}function ye(e){if(e.isRunning)return"Trace executing in an isolated worker. Watchdog armed; UI remains responsive.";if(!e.result?.profile)return e.mode==="edit"?"Patch the kernel. I will flag every stall signature.":"No trace yet. The telemetry grid is quiet.";const t=y[e.currentLevelIndex],n=$(t,e.result.profile,e.result.diagnostics);return e.result.profile.outputMatchesReference?n.passed?"Stall signature eliminated. Queue the next mission before the trace cache cools.":F(e.result.profile)[0]??"Kernel online, but a hidden stall signature is hoarding cycles.":"C buffer compromised. Correctness first, overclock later."}function be(e,t,n){const o=document.createElement("div");o.className="toolbar";const r=k("Patch kernel",t.mode==="edit"?"button-active":"");r.disabled=t.isRunning,r.addEventListener("click",()=>{t.mode="edit",n()});const a=k("Run trace","button-primary");a.disabled=t.isRunning,a.addEventListener("click",()=>{ve(t,n,e)});const l=k("Reset baseline","");l.disabled=t.isRunning,l.addEventListener("click",()=>{t.source=K,t.result=null,t.mode="edit",n()});const s=k("Next mission","");s.disabled=t.isRunning||t.currentLevelIndex>=t.unlockedLevelIndex||t.currentLevelIndex>=y.length-1,s.addEventListener("click",()=>{t.currentLevelIndex<t.unlockedLevelIndex&&t.currentLevelIndex<y.length-1&&(t.currentLevelIndex+=1,n())});const i=k("Abort trace","");i.disabled=!t.isRunning,i.addEventListener("click",()=>{we(t,n,e)});const c=document.createElement("div");c.className="toolbar-spacer";const m=document.createElement("span");m.className="muted",m.textContent=t.isRunning?`${y[t.currentLevelIndex].title} - Trace running, watchdog armed`:`${y[t.currentLevelIndex].title} - ${t.mode==="edit"?"Patch mode":"Trace results"}`,o.append(r,a,i,c,s,l,m);const u=t.mode==="edit"?$e(t):Ae(t);e.replaceChildren(o,u)}function ve(e,t,n){if(e.isRunning)return;const o=e.unlockedLevelIndex,r=e.activeRunId+1,a=new Worker(new URL(""+new URL("runWorker-BXALicKZ.js",import.meta.url).href,import.meta.url),{type:"module"});b?.terminate(),b=a,e.activeRunId=r,e.isRunning=!0,e.runStartedAt=performance.now(),e.result=null,e.mode="run",t();const l=window.setTimeout(()=>{e.activeRunId!==r||!e.isRunning||(a.terminate(),b===a&&(b=null),E(e,t,n,r,_(`Trace watchdog aborted after ${(I/1e3).toFixed(0)} seconds. The kernel may be infinite-looping or too expensive for the browser budget.`),o))},I);a.addEventListener("message",s=>{s.data.runId!==r||e.activeRunId!==r||!e.isRunning||(window.clearTimeout(l),a.terminate(),b===a&&(b=null),E(e,t,n,r,s.data.result,o))}),a.addEventListener("error",s=>{e.activeRunId!==r||!e.isRunning||(window.clearTimeout(l),a.terminate(),b===a&&(b=null),E(e,t,n,r,_(`Trace worker fault: ${s.message}`),o))}),a.postMessage({runId:r,source:e.source})}function we(e,t,n){e.isRunning&&(b?.terminate(),b=null,E(e,t,n,e.activeRunId,_("Trace aborted by operator."),e.unlockedLevelIndex))}function E(e,t,n,o,r,a){if(e.activeRunId!==o)return;e.result=r,e.isRunning=!1,e.runStartedAt=null,Ee(e),e.mode="run",t();const l=xe(e,a);Ce(l),ke(n,l)}function _(e){return{diagnostics:[{line:0,severity:"error",message:e}],profile:null,matrices:null}}function xe(e,t){const n=e.result?.diagnostics.filter(r=>r.severity==="error")??[],o=e.result?.profile;return!o||n.length>0||!o.outputMatchesReference?"fail":e.unlockedLevelIndex>t?"levelup":"pass"}function ke(e,t){e.classList.remove("trace-pulse-pass","trace-pulse-fail","trace-pulse-levelup"),e.offsetWidth,e.classList.add(`trace-pulse-${t}`),window.setTimeout(()=>{e.classList.remove(`trace-pulse-${t}`)},900)}function Ce(e){const t=window.AudioContext??window.webkitAudioContext;if(t)try{const n=new t;(e==="fail"?[146.83,110]:e==="levelup"?[261.63,329.63,493.88,659.25]:[220,329.63,440]).forEach((r,a)=>{Te(n,r,a*.07,e==="levelup"?.13:.1,e)}),window.setTimeout(()=>{n.close()},800)}catch(n){console.warn("CyberGemm sound cue failed:",n)}}function Te(e,t,n,o,r){const a=e.createOscillator(),l=e.createGain(),s=e.currentTime+n,i=s+o;a.type=r==="fail"?"sawtooth":"square",a.frequency.setValueAtTime(t,s),l.gain.setValueAtTime(1e-4,s),l.gain.exponentialRampToValueAtTime(r==="levelup"?.07:.045,s+.01),l.gain.exponentialRampToValueAtTime(1e-4,i),a.connect(l),l.connect(e.destination),a.start(s),a.stop(i+.02)}function Ee(e){const t=e.result?.profile;if(!t)return;$(y[e.currentLevelIndex],t,e.result?.diagnostics??[]).passed&&e.currentLevelIndex===e.unlockedLevelIndex&&(e.unlockedLevelIndex=Math.min(y.length-1,e.unlockedLevelIndex+1))}function $e(e){const t=document.createElement("div");t.className="mode-body";const n=document.createElement("div");n.className="editor-shell";const o=document.createElement("pre");o.className="editor-highlight";const r=document.createElement("code");o.append(r);const a=document.createElement("textarea");a.className="editor",a.spellcheck=!1,a.wrap="off",a.value=e.source,N(r,e.source);let l=!1;const s=document.createElement("div");return s.className="diagnostics",S(s,M(e.source).diagnostics),a.addEventListener("input",()=>{e.source=a.value,N(r,e.source),S(s,M(e.source).diagnostics)}),a.addEventListener("keydown",i=>{if(i.ctrlKey&&!i.shiftKey&&!i.altKey&&!i.metaKey&&i.key.toLowerCase()==="q"){i.preventDefault(),!i.repeat&&l?(e.source=ue(y[e.currentLevelIndex].id),e.result=null,a.value=e.source,N(r,e.source),S(s,M(e.source).diagnostics),l=!1):i.repeat||(l=!0);return}l=!1}),a.addEventListener("scroll",()=>{o.scrollTop=a.scrollTop,o.scrollLeft=a.scrollLeft}),n.append(o,a),t.append(n,s),t}const Le=new Set(["mov","add","sub","mul","div","mod","min","jmp","jz","jnz","jlt","jge","tile.coords","gld.async","gst.async","mma.async","clear","wait","barrier","halt"]),Me=new Set(["coreid","num_cores","total_tiles","m_tiles","n_tiles","k_tiles","tile_m","tile_n","tile_k","matrix_m","matrix_n","matrix_k","all","pad1"]);function N(e,t){e.replaceChildren();const n=t.split(/\r?\n/);n.forEach((o,r)=>{Ne(e,o),r<n.length-1&&e.append(document.createTextNode(`
`))}),t.endsWith(`
`)&&e.append(document.createTextNode(" "))}function Ne(e,t){const n=Be(t),o=n>=0?t.slice(0,n):t,r=n>=0?t.slice(n):"",a=/([A-Za-z_.][\w.]*:|[A-Za-z_.][\w.]*|-?\d+|[,])/g;let l=0,s;for(;(s=a.exec(o))!==null;)s.index>l&&e.append(document.createTextNode(o.slice(l,s.index))),e.append(Se(s[0])),l=s.index+s[0].length;l<o.length&&e.append(document.createTextNode(o.slice(l))),r.length>0&&e.append(g("asm-comment",r))}function Be(e){const t=[e.indexOf("#"),e.indexOf("//"),e.indexOf(";")].filter(n=>n>=0);return t.length>0?Math.min(...t):-1}function Se(e){const t=e.toLowerCase();return e.endsWith(":")?g("asm-label",e):Le.has(t)?g("asm-op",e):/^r\d+$/i.test(e)?g("asm-register",e):/^acc\d+$/i.test(e)?g("asm-accumulator",e):/^[abt]\d+$/i.test(e)?g("asm-slot",e):Me.has(t)?g("asm-constant",e):/^[ABC]$/.test(e)?g("asm-memory",e):/^-?\d+$/.test(e)?g("asm-number",e):g(e===","?"asm-punctuation":"asm-identifier",e)}function g(e,t){const n=document.createElement("span");return n.className=e,n.textContent=t,n}function Ae(e){const t=document.createElement("div");if(t.className="run-panel",e.isRunning){const r=e.runStartedAt?Math.max(0,(performance.now()-e.runStartedAt)/1e3).toFixed(1):"0.0";return t.append(h("Trace running",[Pe([["Status","SIM WORKER ONLINE"],["Watchdog",`${(I/1e3).toFixed(0)}s abort armed`],["Elapsed",`${r}s`],["UI thread","responsive"]]),v("If the kernel infinite-loops or exceeds the browser budget, the worker will be terminated and reported as a trace fault.")])),t}if(!e.result)return t.append(h("Trace awaiting",[v("Press Run trace and watch the stall signatures light up.")])),t;if(!e.result.profile)return t.append(h("Trace rejected",[R(e.result.diagnostics)])),t;const n=e.result.profile;t.append(h("Mission briefing",[_e(e,n)]),h("Daemon chatter",[je(e,n)]),h("Matrix integrity",[H([["Verdict",n.outputMatchesReference?"TRACE CLEAN":"C BUFFER COMPROMISED"],["Corruption delta",n.maxError.toExponential(3)],["Trace cycles",p(n.totalCycles)],["Wall-clock blink",`${n.elapsedMs.toFixed(4)} ms`]])]),h("Telemetry",[We(n)]),h("Stall signatures",[Ge(n)]),h("Trace replay",[Ke(n)]));const o=e.result.diagnostics.filter(r=>r.severity==="error");return o.length>0&&t.append(h("Fault log",[R(o)])),t}function Pe(e){const t=document.createElement("div");t.className="running-trace";const n=document.createElement("div");return n.className="trace-spinner",t.append(n,L(e)),t}function Ie(e,t){const n=document.createElement("ol");return n.className="level-list",y.forEach((o,r)=>{const a=document.createElement("li"),l=document.createElement("button"),s=r>e.unlockedLevelIndex,i=r<e.unlockedLevelIndex;l.type="button",l.disabled=s,l.className=["level-button",r===e.currentLevelIndex?"level-active":"",i?"level-cleared":"",s?"level-locked":""].filter(Boolean).join(" "),l.addEventListener("click",()=>{s||(e.currentLevelIndex=r,t())});const c=document.createElement("span");c.className="level-title",c.textContent=o.title;const m=document.createElement("span");m.className="level-max",m.textContent=`${s?"Sealed - ":""}Target ${o.maxTflops.toFixed(3)} TFLOPS`;const u=document.createElement("span");u.className="level-status",u.textContent=s?"Access sealed":i?"Signature cleared":"Active mission";const f=document.createElement("span");f.className="level-focus",f.textContent=o.bottleneck,l.append(c,u,m,f),a.append(l),n.append(a)}),n}function _e(e,t){const n=y[e.currentLevelIndex],o=$(n,t,e.result?.diagnostics??[]),r=document.createElement("div");return r.className="level-panel",r.append(H([["Mission",n.title],["Throughput",`${t.tflops.toFixed(3)} TFLOPS`],["Target mark",`${n.maxTflops.toFixed(3)} TFLOPS`],["Gate",o.passed?"ACCESS GRANTED":"ACCESS DENIED"],["Signature",o.passed?"Eliminated":"Still active"]]),Re(t,o.passed,n.maxTflops),L([["Signature",n.bottleneck],["Mission",n.objective],["Exploit hint",n.hint],["Verifier",o.status],...o.details.map((a,l)=>[`Trace note ${l+1}`,a])])),r}function Re(e,t,n){const o=document.createElement("div");return o.className="trace-badges",o.append(B("Clean C",e.outputMatchesReference,"Correct C unlocks the trace gate."),B("Stall killed",t,"Eliminate this mission's focused stall signature."),B("Peak chase",e.tflops>=n*.98,"Reach the target TFLOPS mark.")),o}function B(e,t,n){const o=document.createElement("span");return o.className=`trace-badge ${t?"trace-badge-earned":"trace-badge-missing"}`,o.title=n,o.textContent=`${t?"[+]":"[ ]"} ${e}`,o}function je(e,t){const n=y[e.currentLevelIndex],o=$(n,t,e.result?.diagnostics??[]),r=document.createElement("div");r.className=`feedback-panel ${o.passed?"feedback-win":t.outputMatchesReference?"feedback-warn":"feedback-fail"}`;const a=document.createElement("strong");a.textContent=o.passed?"Optimization unlocked!":t.outputMatchesReference?"Kernel accepted.":"Matrix corruption detected.";const l=v(Oe(t,o.passed));return r.append(a,l),r}function Oe(e,t){if(!e.outputMatchesReference)return"The output buffer is corrupted. No overclock credit until C matches the reference trace.";if(t)return`Pipeline breached. ${e.tflops.toFixed(3)} TFLOPS and the focused stall signature is gone.`;const n=F(e)[0];return n?`Kernel accepted, but listen: ${n}`:"Kernel accepted, but the trace still hides a bottleneck."}function S(e,t){if(t.length===0){e.textContent="No parser diagnostics.";return}e.replaceChildren(R(t))}function h(e,t){const n=document.createElement("section");n.className="section";const o=document.createElement(e==="CyberGemm"?"h1":"h2");return o.textContent=e,n.append(o,...t),n}function v(e){const t=document.createElement("p");return t.className="muted",t.textContent=e,t}function L(e){const t=document.createElement("ul");t.className="metric-list";for(const[n,o]of e){const r=document.createElement("li");r.className="metric-row";const a=document.createElement("span");a.textContent=n;const l=document.createElement("span");l.textContent=o,r.append(a,l),t.append(r)}return t}function Fe(e){const t=document.createElement("table");t.className="compact-table";const n=document.createElement("tbody");for(let o=0;o<e.length;o+=2){const r=document.createElement("tr");r.append(W(e[o])),e[o+1]&&r.append(W(e[o+1])),n.append(r)}return t.append(n),t}function W(e){const t=document.createElement("td"),n=document.createElement("span");n.className="compact-label",n.textContent=e[0];const o=document.createElement("strong");return o.textContent=e[1],t.append(n,o),t}function ze(){const e=document.createElement("ul");e.className="instruction-list";for(const t of q){const n=document.createElement("li");n.className="instruction";const o=document.createElement("h3"),r=document.createElement("code");r.textContent=t.syntax,o.append(r);const a=v(t.description);n.append(o,a),e.append(n)}return e}function k(e,t){const n=document.createElement("button");return n.className=["button",t].filter(Boolean).join(" "),n.type="button",n.textContent=e,n}function H(e){const t=document.createElement("div");t.className="grid";for(const[n,o]of e){const r=document.createElement("div");r.className="card";const a=document.createElement("p");a.className="card-label",a.textContent=n;const l=document.createElement("p");l.className="card-value",l.textContent=o,r.append(a,l),t.append(r)}return t}function R(e){const t=document.createElement("div");t.className="metric-list";for(const n of e){const o=document.createElement("div");o.className=`metric-row diagnostic-${n.severity}`;const r=document.createElement("span");r.textContent=n.line>0?`Line ${n.line}`:"Runtime";const a=document.createElement("span");a.textContent=n.message,o.append(r,a),t.append(o)}return t}function We(e){const t=e.coreProfiles.reduce((s,i)=>s+i.globalWaitCycles,0),n=e.coreProfiles.reduce((s,i)=>s+i.tensorWaitCycles,0),o=e.coreProfiles.reduce((s,i)=>s+i.barrierWaitCycles,0),r=e.coreProfiles.reduce((s,i)=>s+i.bankConflictCycles,0),a=e.coreProfiles.filter(s=>s.tilesCompleted>0).length,l=e.tensorProfiles.filter(s=>s.operations>0).length;return L([["TFLOPS output",e.tflops.toFixed(3)],["HBM bandwidth",`${e.globalBandwidthGbps.toFixed(2)} GB/s`],["Tensor array uptime",O(e.tensorUtilization)],["Core grid uptime",O(e.computeUtilization)],["HBM stall cycles",`${p(t)} cycles`],["Tensor stall cycles",`${p(n)} cycles`],["Sync stall cycles",`${p(o)} cycles`],["Bank collision tax",`${p(r)} cycles`],["Global bytes traced",`${p(e.globalBytesMoved)} B`],["Active core lanes",`${a} / ${e.coreProfiles.length}`],["Tensor units online",`${l} / ${e.tensorProfiles.length}`]])}function Ge(e){const t=F(e),n=document.createElement("ul");n.className="hotspot-list";for(const o of t){const r=document.createElement("li");r.className="hotspot",r.textContent=o,n.append(r)}return n}function F(e){const t=e.coreProfiles.filter(i=>i.tilesCompleted>0),n=e.tensorProfiles.filter(i=>i.operations>0),o=e.coreProfiles.reduce((i,c)=>i+c.globalWaitCycles,0),r=e.coreProfiles.reduce((i,c)=>i+c.tensorWaitCycles,0),a=e.coreProfiles.reduce((i,c)=>i+c.bankConflictCycles,0),l=e.coreProfiles.flatMap(i=>i.timeline.filter(c=>c.kind==="wait")),s=[];return t.length<e.coreProfiles.length/4&&s.push(`Dormant cores detected: only ${t.length} of ${e.coreProfiles.length} compute cores wrote tiles. Distribute tile ownership across the grid.`),e.tensorUtilization<.1&&s.push(`Tensor array underfed (${O(e.tensorUtilization)} uptime). Overlap loads and issue more MMA work before waiting.`),o>r*2&&s.push(`HBM traffic jam: global/store waits cost ${p(o)} cycles because the kernel waits too eagerly.`),a>0&&s.push(`Bank collision storm adds ${p(a)} penalty cycles. Use the pad1 layout route for A/B tiles.`),l.length>0&&l.every(i=>i.waitingOn.length===1)&&s.push("Serial wait pattern: each wait watches one async op, so memory and tensor work stay in single-file trace phases."),n.length<e.tensorProfiles.length/4&&s.push(`Tensor occupancy low: ${n.length} of ${e.tensorProfiles.length} units did work.`),s.length>0?s:["No obvious stall signature detected. Trace looks clean."]}function Ke(e){const t=e.coreProfiles.filter(s=>s.timeline.length>0).slice(0,5),n=new Map(t.map(s=>[s.coreId,Ue(s.timeline,6)])),o=[...n.values()].flat(),r=Math.max(1,...o.map(s=>s.endCycle)),a=e.tensorProfiles.flatMap(s=>s.timeline),l=document.createElement("div");if(l.className="visual-timeline",l.append(He()),t.length===0)return l.prepend(v("No waits were recorded. Either the trace was empty or the kernel faulted early.")),l;l.prepend(v(`Trace replay from cores ${t.map(s=>s.coreId).join(", ")}. Bars are active packets; down arrows mark stall start and up arrows mark stall release.`));for(const s of t){const i=document.createElement("div");i.className="timeline-core-group";const c=document.createElement("h3");c.textContent=`Core ${s.coreId} trace lane`;const m=n.get(s.coreId)??[],u=A(e.memoryTimeline.filter(x=>x.coreId===s.coreId),r),f=A(a.filter(x=>x.coreId===s.coreId),r),w=A(e.mathTimeline.filter(x=>x.coreId===s.coreId),r);i.append(c,P("HBM packets",u,r,"memory",De,m),P("Tensor pulses",f,r,"tensor",qe,m),P("Scalar ops",w,r,"math",Ve,m)),l.append(i)}return l}function Ue(e,t){return[...e].sort((n,o)=>n.startCycle-o.startCycle).slice(0,t)}function A(e,t){return[...e].filter(n=>n.startCycle<=t).sort((n,o)=>n.startCycle-o.startCycle)}function P(e,t,n,o,r,a){const l=document.createElement("div");l.className="visual-lane";const s=document.createElement("span");s.className="visual-lane-label",s.textContent=e;const i=document.createElement("div");i.className="visual-track",t.forEach((c,m)=>{const u=document.createElement("span");u.className=`visual-bar visual-${o}`,u.style.left=`${j(c.startCycle,n)}%`,u.style.width=`${Math.max(.5,j(c.endCycle-c.startCycle,n))}%`,u.style.setProperty("--bar-delay",`${m%6*45}ms`),u.title=r(c),i.append(u)});for(const c of a)i.append(G("down",c.startCycle,n,`stall starts: ${c.instruction}`)),i.append(G("up",c.endCycle,n,`stall releases after ${p(c.endCycle-c.startCycle)} cycles`));return l.append(s,i),l}function G(e,t,n,o){const r=document.createElement("span");return r.className=`wait-arrow wait-arrow-${e}`,r.style.left=`${j(t,n)}%`,r.title=o,r.textContent=e==="down"?"↓":"↑",r}function He(){const e=document.createElement("div");return e.className="timeline-legend",e.append(T("HBM packets","visual-memory"),T("Tensor pulses","visual-tensor"),T("Scalar ops","visual-math"),T("↓ stall start / ↑ stall release","visual-wait")),e}function T(e,t){const n=document.createElement("span"),o=document.createElement("span");return o.className=`legend-swatch ${t}`,n.append(o,document.createTextNode(e)),n}function De(e){return`${e.label}: ${p(e.startCycle)}-${p(e.endCycle)} cycles, ${p(e.bytes)} B`}function qe(e){return`${e.tile} on tensor ${e.unitId}: ${p(e.startCycle)}-${p(e.endCycle)} cycles`}function Ve(e){return`${e.instruction}: ${p(e.startCycle)}-${p(e.endCycle)} cycles`}function j(e,t){return Math.max(0,Math.min(100,e/t*100))}function p(e){return Math.round(e).toLocaleString()}function O(e){return`${(e*100).toFixed(1)}%`}const D=document.querySelector("#app");if(!D)throw new Error("Missing #app root element");he(D);
