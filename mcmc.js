// Simulation and MCMC engine
class Node {
    constructor(id) {
        this.id = id;
        this.childs = [];
        this.parent = null;
        this.length = 0;
        this.height = 0;
        this.seq = "";
    }
}

// ─── FASTA parser ─────────────────────────────────────────────────────────────
function parseFasta(text) {
    const msa = {};
    let currentName = null;
    for (const rawLine of text.split(/\r?\n/)) {
        const line = rawLine.trim();
        if (!line) continue;
        if (line.startsWith('>')) {
            currentName = line.slice(1).trim().split(/\s+/)[0]; // first word only
            msa[currentName] = '';
        } else if (currentName) {
            msa[currentName] += line.toUpperCase().replace(/[^ACGT]/g, 'A'); // treat ambiguous as A
        }
    }
    return msa;
}

// Build a random binary tree purely from an array of leaf name strings.
function buildRandomTreeFromNames(names) {
    let nodeId = 0;
    let pending = names.map(name => {
        const n = new Node(name);
        n.length = Math.random() * 0.5 + 0.05;
        return n;
    });
    while (pending.length > 1) {
        let i = Math.floor(Math.random() * pending.length);
        let j; do { j = Math.floor(Math.random() * pending.length); } while (j === i);
        if (i > j) [i, j] = [j, i];
        const a = pending[i], b = pending[j];
        pending.splice(j, 1); pending.splice(i, 1);
        const internal = new Node('_r' + (nodeId++));
        internal.length = Math.random() * 0.5 + 0.05;
        a.parent = internal; b.parent = internal;
        internal.childs = [a, b];
        pending.push(internal);
    }
    const root = pending[0];
    root.length = 0; root.parent = null;
    return root;
}

// Global state
let state = {
    trueTree: null,
    trueBirth: NaN,
    trueDeath: NaN,
    trueMu: NaN,
    trueRootHeight: 0,
    isRealData: false,
    msa: {},
    mcmcRunning: false,
    mcmcStep: 0,
    accepted: 0,
    currentTree: null,
    currentBirth: 1.0,
    currentDeath: 0.5,
    currentMu: 0.01,
    currentLogPosterior: -Infinity,
    treeSamples: [],          // ring buffer of cloned trees for DensiTree
    traceData: {
        gen: [],
        logPost: [],
        birth: [],
        death: [],
        mu: []
    }
};

let chart, histBirthChart, histDeathChart, histMuChart;

function makeHistData(samples, nBins, color) {
    if (!samples || samples.length === 0) return { labels: [], data: [] };
    const min = Math.max(0, Math.min(...samples));
    const max = Math.max(...samples);
    const range = max - min || 1;
    const binW = range / nBins;
    const counts = new Array(nBins).fill(0);
    for (const v of samples) {
        let i = Math.floor((v - min) / binW);
        if (i >= nBins) i = nBins - 1;
        counts[i]++;
    }
    const labels = counts.map((_, i) => (min + (i + 0.5) * binW).toFixed(2));
    return { labels, data: counts };
}

// Returns a Chart.js plugin that draws a dashed vertical line at the true parameter value.
function makeVertLinePlugin(getTrueValue, strokeStyle = '#000') {
    return {
        id: 'trueLine',
        afterDraw(chart) {
            const trueVal = getTrueValue();
            if (!isFinite(trueVal)) return;
            const labels = chart.data.labels;
            if (!labels || labels.length < 2) return;
            const vals = labels.map(Number);
            const xMin = vals[0];
            const xMax = vals[vals.length - 1];
            if (xMax === xMin) return;
            const frac = (trueVal - xMin) / (xMax - xMin);
            const xScale = chart.scales.x;
            const x = xScale.left + frac * (xScale.right - xScale.left);
            const ctx = chart.ctx;
            ctx.save();
            ctx.beginPath();
            ctx.moveTo(x, chart.scales.y.top);
            ctx.lineTo(x, chart.scales.y.bottom);
            ctx.strokeStyle = strokeStyle;
            ctx.lineWidth = 2;
            ctx.setLineDash([6, 3]);
            ctx.stroke();
            ctx.restore();
        }
    };
}

function initHistChart(canvasId, label, color, getTrueValue) {
    const plugins = getTrueValue ? [makeVertLinePlugin(getTrueValue)] : [];
    const ctx = document.getElementById(canvasId).getContext('2d');
    return new Chart(ctx, {
        type: 'bar',
        plugins,
        data: {
            labels: [],
            datasets: [{
                label,
                data: [],
                backgroundColor: color,
                borderWidth: 0,
                barPercentage: 1.0,
                categoryPercentage: 1.0
            }]
        },
        options: {
            animation: false,
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    title: { display: true, text: label },
                    ticks: { maxTicksLimit: 6 }
                },
                y: {
                    title: { display: true, text: 'Count' },
                    beginAtZero: true
                }
            }
        }
    });
}

function updateHistograms() {
    const nBins = 30;
    // Use only the last half of the chain as the posterior (50% burn-in)
    const burnIn = Math.floor(state.traceData.gen.length / 2);
    const bSamples = state.traceData.birth.slice(burnIn);
    const dSamples = state.traceData.death.slice(burnIn);

    const bH = makeHistData(bSamples, nBins);
    histBirthChart.data.labels = bH.labels;
    histBirthChart.data.datasets[0].data = bH.data;
    histBirthChart.update();

    const dH = makeHistData(dSamples, nBins);
    histDeathChart.data.labels = dH.labels;
    histDeathChart.data.datasets[0].data = dH.data;
    histDeathChart.update();

    const muSamples = state.traceData.mu.slice(burnIn);
    const mH = makeHistData(muSamples, nBins);
    histMuChart.data.labels = mH.labels;
    histMuChart.data.datasets[0].data = mH.data;
    histMuChart.update();
}

function initChart() {
    const ctx = document.getElementById('traceChart').getContext('2d');
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'β (Birth)',
                    data: [],
                    borderColor: 'blue',
                    borderWidth: 1,
                    pointRadius: 0,
                    yAxisID: 'y',
                    fill: false,
                    tension: 0
                },
                {
                    label: 'δ (Death)',
                    data: [],
                    borderColor: 'red',
                    borderWidth: 1,
                    pointRadius: 0,
                    yAxisID: 'y',
                    fill: false,
                    tension: 0
                },
                {
                    label: 'μ (Mutation ×10)',
                    data: [],
                    borderColor: 'green',
                    borderWidth: 1,
                    pointRadius: 0,
                    yAxisID: 'y1',
                    fill: false,
                    tension: 0
                }
            ]
        },
        options: {
            animation: false,
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: true, labels: { boxWidth: 12, font: { size: 11 } } } },
            scales: {
                x: { title: { display: true, text: 'Generation' } },
                y: { title: { display: true, text: 'β / δ' }, min: 0, position: 'left' },
                y1: { title: { display: true, text: 'μ' }, min: 0, position: 'right',
                      grid: { drawOnChartArea: false } }
            }
        }
    });

    histBirthChart = initHistChart('histBirthChart', 'β (Birth)',        'rgba(0,0,200,0.5)',   () => state.trueBirth);
    histDeathChart = initHistChart('histDeathChart', 'δ (Death)',        'rgba(200,0,0,0.5)',   () => state.trueDeath);
    histMuChart    = initHistChart('histMuChart',    'μ (Mutation Rate)','rgba(0,150,0,0.5)',   () => state.trueMu);
}

function parseNewick(node) {
    if (!node) return "";
    if (node.childs.length === 0) return node.id + ":" + node.length.toFixed(4);
    let nwk = "(" + node.childs.map(c => parseNewick(c)).join(",") + ")";
    if (node.length !== undefined) nwk += ":" + node.length.toFixed(4);
    return nwk;
}

// Generate Birth-Death Tree
function simulateBD(lambda, mu, nTaxa = 15) {
    let t = 0;
    let leaves = [new Node("root")];
    let nextId = 1;

    while(leaves.length < nTaxa) {
        let n = leaves.length;
        let totalRate = n * (lambda + mu);
        let dt = -Math.log(Math.random()) / totalRate;
        t += dt;

        for(let l of leaves) l.length += dt;

        let r = Math.random();
        if(r < lambda / (lambda + mu)) {
            // Speciation
            let targetIdx = Math.floor(Math.random() * leaves.length);
            let target = leaves[targetIdx];
            let c1 = new Node("N" + (nextId++));
            let c2 = new Node("N" + (nextId++));
            
            c1.parent = target;
            c2.parent = target;
            target.childs = [c1, c2];
            
            leaves.splice(targetIdx, 1, c1, c2);
        } else {
            // Extinction
            let targetIdx = Math.floor(Math.random() * leaves.length);
            if(leaves.length > 1) {
                let target = leaves[targetIdx];
                // Dead end
                target.leaf_dead = true;
                leaves.splice(targetIdx, 1);
            }
        }
    }
    
    // Add extra branch length to align extant tips
    let maxLen = 0;
    for(let l of leaves) maxLen = Math.max(maxLen, getDepth(l));
    for(let l of leaves) l.length += (maxLen - getDepth(l));

    // Filter to tree to only leaves that are alive (or just name nodes properly)
    assignExtantNames(leaves);
    
    // Retain full root
    let root = leaves[0];
    while(root.parent) root = root.parent;
    return root;
}

function getDepth(node) {
    let d = 0;
    let c = node;
    while(c) { d += c.length; c = c.parent; }
    return d;
}

function assignExtantNames(leaves) {
    let leafCount = 1;
    for(let i=0; i<leaves.length; i++) {
        leaves[i].id = "T" + (leafCount++);
    }
}

const NUC = ["A", "C", "G", "T"];
function mutateSeq(seq, time, rate=0.01) {
    let newSeq = "";
    for(let i=0; i<seq.length; i++) {
        if(Math.random() < 1 - Math.exp(-rate * time)) {
            newSeq += NUC[Math.floor(Math.random()*4)]; // Simple JC
        } else {
            newSeq += seq[i];
        }
    }
    return newSeq;
}

function evolveSequences(node, seqLen, mu) {
    if(!node.parent) {
        node.seq = Array(seqLen).fill(0).map(()=>NUC[Math.floor(Math.random()*4)]).join('');
    } else {
        node.seq = mutateSeq(node.parent.seq, node.length, mu);
    }
    for(let c of node.childs) {
        evolveSequences(c, seqLen, mu);
    }
}

function extractMSA(node, msa) {
    if(node.childs.length === 0 && !node.leaf_dead) {
        msa[node.id] = node.seq;
    }
    for(let c of node.childs) extractMSA(c, msa);
}

function cloneTree(node, parent = null) {
    let copy = new Node(node.id);
    copy.length = node.length;
    copy.height = node.height;
    copy.seq = node.seq;
    copy.leaf_dead = node.leaf_dead;
    copy.parent = parent;
    for(let c of node.childs) {
        copy.childs.push(cloneTree(c, copy));
    }
    return copy;
}

// ─── Real Felsenstein JC Pruning Likelihood ──────────────────────────────────

const NUC_IDX = {A:0, C:1, G:2, T:3};

// JC transition probability: P(same | t, mu) or P(diff | t, mu)
// Here `mu` is the per-site substitution rate (= 4α/3 in standard JC notation).
// This matches the simulation in mutateSeq where the same `rate` parameter is used.
function jcProb(isSame, t, mu) {
    const p = Math.exp(-mu * t);
    return isSame ? 0.25 + 0.75 * p : 0.25 - 0.25 * p;
}

// Felsenstein pruning at one site — returns Float64Array[4] of partial likelihoods
function pruningNode(node, siteCols, mu) {
    if (!node.childs || node.childs.length === 0) {
        const L = new Float64Array(4);
        const idx = NUC_IDX[siteCols[node.id]];
        if (idx !== undefined) L[idx] = 1.0;
        else L.fill(1.0); // ambiguous / gap
        return L;
    }
    const L = new Float64Array([1, 1, 1, 1]);
    for (const child of node.childs) {
        if (child.leaf_dead) continue;
        const cL = pruningNode(child, siteCols, mu);
        const t = Math.max(child.length, 1e-6);
        const pS = jcProb(true,  t, mu);
        const pD = jcProb(false, t, mu);
        const sumC = cL[0] + cL[1] + cL[2] + cL[3];
        // For each parent state i: sum_j P(j|i) * cL[j] = pD*sumC + (pS-pD)*cL[i]
        for (let i = 0; i < 4; i++) {
            L[i] *= pD * sumC + (pS - pD) * cL[i];
        }
    }
    return L;
}

// Sum over all sites — returns log-likelihood
function computeFelsensteinLogLikelihood(tree, msa, mu) {
    const taxa = Object.keys(msa);
    if (!taxa.length) return 0;
    const seqLen = msa[taxa[0]].length;
    let logL = 0;
    for (let site = 0; site < seqLen; site++) {
        const siteCols = {};
        for (const tx of taxa) siteCols[tx] = msa[tx][site];
        const rootL = pruningNode(tree, siteCols, mu);
        // Equal root frequencies under JC (0.25 each)
        const siteL = 0.25 * (rootL[0] + rootL[1] + rootL[2] + rootL[3]);
        if (siteL <= 0) return -Infinity;
        logL += Math.log(siteL);
    }
    return logL;
}

// ─── MCMC Convergence Diagnostics ────────────────────────────────────────────

// Effective Sample Size via integrated autocorrelation time
function computeESS(samples) {
    const n = samples.length;
    if (n < 20) return n;
    const mean = samples.reduce((a, b) => a + b, 0) / n;
    const variance = samples.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
    if (variance < 1e-12) return 1;
    let rhoSum = 0;
    const maxLag = Math.min(n - 1, 200);
    for (let lag = 1; lag <= maxLag; lag++) {
        let cov = 0;
        for (let i = 0; i < n - lag; i++) cov += (samples[i] - mean) * (samples[i + lag] - mean);
        const rho = cov / (n * variance);
        if (rho < 0.05) break;
        rhoSum += rho;
    }
    return Math.min(n, Math.round(n / (1 + 2 * rhoSum)));
}

// Shortest interval containing credMass of the sorted posterior samples
function computeHPD(samples, credMass = 0.95) {
    if (samples.length < 10) return [NaN, NaN];
    const sorted = [...samples].sort((a, b) => a - b);
    const n = sorted.length;
    const nIn = Math.floor(credMass * n);
    let bestWidth = Infinity, lo = sorted[0], hi = sorted[n - 1];
    for (let i = 0; i <= n - nIn; i++) {
        const w = sorted[i + nIn - 1] - sorted[i];
        if (w < bestWidth) { bestWidth = w; lo = sorted[i]; hi = sorted[i + nIn - 1]; }
    }
    return [lo, hi];
}

// Build a completely random binary tree from the same leaf names as the true tree.
// This gives a genuinely random starting topology, not just scrambled branch lengths.
function buildRandomTree(trueRoot) {
    // Collect only extant (non-dead) leaf names
    let leafNames = [];
    function collectLeaves(n) {
        if (!n.childs || n.childs.length === 0) {
            if (!n.leaf_dead) leafNames.push(n.id);
        } else for (let c of n.childs) collectLeaves(c);
    }
    collectLeaves(trueRoot);

    // Shuffle leaf names
    for (let i = leafNames.length - 1; i > 0; i--) {
        let j = Math.floor(Math.random() * (i + 1));
        [leafNames[i], leafNames[j]] = [leafNames[j], leafNames[i]];
    }

    // Build random binary tree by repeatedly pairing
    let nodeId = 0;
    let pending = leafNames.map(name => {
        let n = new Node(name);
        n.length = Math.random() * 2 + 0.1; // random branch length
        return n;
    });

    while (pending.length > 1) {
        // Pick two at random and merge under a new internal node
        let i = Math.floor(Math.random() * pending.length);
        let j;
        do { j = Math.floor(Math.random() * pending.length); } while (j === i);
        if (i > j) [i, j] = [j, i]; // ensure i < j for splice order

        let a = pending[i];
        let b = pending[j];
        pending.splice(j, 1);
        pending.splice(i, 1);

        let internal = new Node('_r' + (nodeId++));
        internal.length = Math.random() * 2 + 0.1;
        a.parent = internal;
        b.parent = internal;
        internal.childs = [a, b];
        pending.push(internal);
    }

    let root = pending[0];
    root.length = 0;
    root.parent = null;
    return root;
}

// Collect the set of leaf IDs below a node
function getLeafSet(node) {
    let leaves = new Set();
    function traverse(n) {
        if (!n.childs || n.childs.length === 0) leaves.add(n.id);
        for (let c of n.childs) traverse(c);
    }
    traverse(node);
    return leaves;
}

// Compute all bipartition splits for a tree (as sorted comma-joined leaf strings)
function getSplits(root) {
    let splits = new Set();
    function traverse(n) {
        if (n !== root && n.childs && n.childs.length > 0) {
            let key = [...getLeafSet(n)].sort().join(',');
            splits.add(key);
        }
        for (let c of n.childs) traverse(c);
    }
    traverse(root);
    return splits;
}

// Robinson-Foulds style distance (shared splits) + branch length penalty
function getTreeDistance(t1, t2) {
    if (!t1 || !t2) return 0;
    
    let splits1 = getSplits(t1);
    let splits2 = getSplits(t2);
    
    let rfDist = 0;
    for (let s of splits1) if (!splits2.has(s)) rfDist++;
    for (let s of splits2) if (!splits1.has(s)) rfDist++;
    
    return rfDist; // purely topology-based distance
}

// ----------------------------------------------------
// UI Logic

document.addEventListener("DOMContentLoaded", () => {
    initChart();
    
    document.getElementById("btn-simulate").addEventListener("click", () => {
        state.isRealData = false;
        state.trueBirth = parseFloat(document.getElementById("sim-birth").value);
        state.trueDeath = parseFloat(document.getElementById("sim-death").value);
        let seqL = parseInt(document.getElementById("seq-length").value);

        state.trueMu   = parseFloat(document.getElementById("true-mu").value) || 0.01;
        state.trueTree = simulateBD(state.trueBirth, state.trueDeath, 20);
        evolveSequences(state.trueTree, seqL, state.trueMu);
        state.trueRootHeight = computeRootHeight(state.trueTree);
        document.getElementById('fbd-root-age').textContent = state.trueRootHeight.toFixed(3);
        
        state.msa = {};
        extractMSA(state.trueTree, state.msa);
        
        let msaStr = Object.keys(state.msa).map(k => ">" + k + "\n" + state.msa[k].substring(0,50)).join("\n");
        document.getElementById("alignment-display").textContent = msaStr;
        document.getElementById("data-container").style.display = "block";

        document.getElementById("btn-mcmc").disabled = false;
        
        // Init MCMC starting state with a completely random tree topology
        state.currentTree = buildRandomTree(state.trueTree);
        state.currentBirth = Math.random() * 3 + 0.5;       // Random start
        state.currentDeath = Math.random() * 1.5 + 0.1;     // Random start
        state.currentMu    = Math.random() * 0.08 + 0.002;  // Random start
        state.mcmcRunning = false;
        
        drawTreeCanvas(state.trueTree, "true-tree-container");
        drawTreeCanvas(state.currentTree, "tree-container");
        
        let startDist = getTreeDistance(state.currentTree, state.trueTree);
        document.getElementById("tree-dist").textContent = `Dist: ${startDist.toFixed(2)}`;
        
        document.getElementById("recovery-panel").innerHTML = `
            <table class="recovery-table">
                <tr><th>Param</th><th>True</th><th>Mean</th><th>95% HPD</th><th>ESS</th></tr>
                <tr><td>β</td><td>${state.trueBirth.toFixed(3)}</td><td>—</td><td>—</td><td>—</td></tr>
                <tr><td>δ</td><td>${state.trueDeath.toFixed(3)}</td><td>—</td><td>—</td><td>—</td></tr>
                <tr><td>μ</td><td>${state.trueMu.toFixed(4)}</td><td>—</td><td>—</td><td>—</td></tr>
            </table>
            <div style="font-size: 11px; color: #777; margin-top: 8px;">T, ρ, ψ, r fixed. ¼ burn-in discarded.</div>
        `;

    });

    // ── FASTA file upload ──────────────────────────────────────────────────
    document.getElementById("fasta-file-input").addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (!file) return;
        document.getElementById("fasta-filename").textContent = file.name;
    });

    document.getElementById("fasta-file-input").addEventListener("change", (e) => {
        const file = e.target.files[0];
        const display = document.getElementById("fasta-filename");
        display.textContent = file ? file.name : "no file chosen";
    });

    document.getElementById("btn-load-fasta").addEventListener("click", () => {
        const file = document.getElementById("fasta-file-input").files[0];
        if (!file) { alert("Please choose a FASTA file first."); return; }

        const reader = new FileReader();
        reader.onload = (e) => {
            const msa = parseFasta(e.target.result);
            const names = Object.keys(msa);
            if (names.length < 4) { alert("FASTA must contain at least 4 sequences."); return; }
            const seqLen = msa[names[0]].length;
            if (!names.every(n => msa[n].length === seqLen)) {
                alert("All sequences must be the same length (aligned FASTA)."); return;
            }

            // Validate root age calibration
            const rootAgeEl = document.getElementById("fasta-root-age");
            const rootAge = parseFloat(rootAgeEl.value);
            if (!(rootAge > 0)) { alert("Please enter a positive root age calibration."); return; }

            state.isRealData  = true;
            state.trueTree    = null;
            state.trueBirth   = NaN;
            state.trueDeath   = NaN;
            state.trueMu      = NaN;
            state.trueRootHeight = rootAge;
            state.msa         = msa;

            state.currentTree  = buildRandomTreeFromNames(names);
            state.currentBirth = Math.random() * 3 + 0.5;
            state.currentDeath = Math.random() * 1.5 + 0.1;
            state.currentMu    = Math.random() * 0.08 + 0.002;
            state.mcmcRunning  = false;

            document.getElementById('fbd-root-age').textContent = rootAge.toFixed(3);

            // Show alignment preview
            let msaStr = names.map(k => ">" + k + "\n" + msa[k].substring(0, 60)).join("\n");
            document.getElementById("alignment-display").textContent = msaStr;
            document.getElementById("data-container").style.display = "block";

            // Update panels
            const trueTreeContainer = document.getElementById("true-tree-container");
            trueTreeContainer.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#aaa;font-size:13px;font-family:sans-serif;">No true tree — real data mode</div>`;
            drawTreeCanvas(state.currentTree, "tree-container");
            document.getElementById("tree-dist").textContent = "";

            document.getElementById("recovery-panel").innerHTML = `
                <table class="recovery-table">
                    <tr><th>Param</th><th>Mean</th><th>95% HPD</th><th>ESS</th></tr>
                    <tr><td>β</td><td>—</td><td>—</td><td>—</td></tr>
                    <tr><td>δ</td><td>—</td><td>—</td><td>—</td></tr>
                    <tr><td>μ</td><td>—</td><td>—</td><td>—</td></tr>
                </table>
                <div style="font-size:11px;color:#777;margin-top:8px;">${names.length} taxa · ${seqLen} bp</div>
            `;

            document.getElementById("btn-mcmc").disabled = false;
        };
        reader.readAsText(file);
    });

    document.getElementById("btn-mcmc").addEventListener("click", () => {
        state.mcmcRunning = true;
        document.getElementById("btn-mcmc").style.display = "none";
        document.getElementById("btn-stop").style.display = "block";
        
        state.mcmcStep = 0;
        state.accepted = 0;
        state.traceData = { gen: [], logPost: [], birth: [], death: [], mu: [] };
        state.treeSamples = [];
        
        state.currentLogPosterior = computeLogPosterior(
            state.currentBirth, state.currentDeath, state.currentMu, state.currentTree);
        
        runMCMCStep();
    });

    document.getElementById("btn-stop").addEventListener("click", () => {
        state.mcmcRunning = false;
        document.getElementById("btn-mcmc").style.display = "block";
        document.getElementById("btn-stop").style.display = "none";
    });
});

// Collect all branch lengths in a tree
function collectBranchLengths(node, out = []) {
    if (node.parent) out.push(node.length || 0);
    for (let c of node.childs) collectBranchLengths(c, out);
    return out;
}

// Max root-to-tip path (= root age in branch-length units)
function computeRootHeight(tree) {
    function maxPath(node) {
        if (!node.childs || node.childs.length === 0) return node.leaf_dead ? -Infinity : 0;
        let best = -Infinity;
        for (const c of node.childs) {
            const h = (c.length || 0) + maxPath(c);
            if (h > best) best = h;
        }
        return Math.max(best, 0);
    }
    return maxPath(tree);
}

// ─── Birth-Death Tree Likelihood ─────────────────────────────────────────────
// Yule approximation using net diversification rate r = beta - delta.
// Collects internal node heights (time from present = 0) and scores them under
// the Yule waiting-time density.  This is the primary constraint on β and δ.
function computeBDTreeLogLik(tree, beta, delta) {
    const r = beta - delta;
    if (r <= 1e-6) return -Infinity; // require positive net diversification

    // Compute height of a node as max path-length to any live descendant tip
    function nodeHeight(node) {
        if (!node.childs || node.childs.length === 0) return 0;
        let best = 0;
        for (const c of node.childs) {
            if (!c.leaf_dead) {
                const h = (c.length || 0) + nodeHeight(c);
                if (h > best) best = h;
            }
        }
        return best;
    }

    // Collect heights of every internal node (including root)
    const heights = [];
    function collectInternal(node) {
        if (node.childs && node.childs.length > 0 && !node.leaf_dead) {
            heights.push(nodeHeight(node));
            for (const c of node.childs) if (!c.leaf_dead) collectInternal(c);
        }
    }
    collectInternal(tree);

    if (heights.length < 2) return 0;
    heights.sort((a, b) => b - a); // descending: [root age, ..., youngest]

    // Yule log L: for each depth interval i, there are (i+2) lineages.
    // Waiting time density: Exp((i+2)*r)  →  log contribution = log((i+2)*r) - (i+2)*r*gap
    let logL = 0;
    for (let i = 0; i < heights.length; i++) {
        const lineages = i + 2;
        const gap = heights[i] - (i + 1 < heights.length ? heights[i + 1] : 0);
        logL += Math.log(lineages * r) - lineages * r * gap;
    }
    return logL;
}

// Log-posterior: Felsenstein likelihood + BD tree likelihood + priors.
function computeLogPosterior(b, d, mu, tree) {
    if (b <= 0 || d < 0 || mu <= 0) return -Infinity;

    // Log-normal priors on rates  (log P ∝ -(log θ - log mean)²)
    const priorBMean  = parseFloat(document.getElementById("prior-birth-mean").value);
    const priorDMean  = parseFloat(document.getElementById("prior-death-mean").value);
    const priorMuMean = parseFloat(document.getElementById("prior-mu-mean").value);
    const logPrior = -Math.pow(Math.log(b)  - Math.log(priorBMean),  2)
                   - Math.pow(Math.log(d)  - Math.log(priorDMean),  2)
                   - Math.pow(Math.log(mu) - Math.log(priorMuMean), 2);

    // Exponential prior on every branch length (rate = 2, mean = 0.5)
    const expRate = 2;
    let branchLengthLogPrior = 0;
    for (const bl of collectBranchLengths(tree)) {
        if (bl <= 0) return -Infinity;
        branchLengthLogPrior += Math.log(expRate) - expRate * bl;
    }

    // Real Felsenstein pruning likelihood from the sequence alignment (informs μ + branch lengths)
    const logLikelihood = computeFelsensteinLogLikelihood(tree, state.msa, mu);

    // Birth-death tree likelihood from branching times (informs β, δ)
    const logBDTree = computeBDTreeLogLik(tree, b, d);

    // Root-age calibration prior — anchors the time scale, breaking the μ↔β confound.
    // Analogous to a fossil calibration in BEAST: the root age should match the observed
    // tree height to within ~5%.  Without this, μ and β are non-identifiable.
    const sampledRootHeight = computeRootHeight(tree);
    const rootSigmaFrac = 0.05; // allow ±5% of true root age
    const rootSigma = rootSigmaFrac * state.trueRootHeight;
    const logRootCalib = rootSigma > 0
        ? -0.5 * Math.pow((sampledRootHeight - state.trueRootHeight) / rootSigma, 2)
        : 0;

    return logLikelihood + logBDTree + logPrior + branchLengthLogPrior + logRootCalib;
}

// NNI move: pick an internal edge (v with internal parent u),
// then swap one child of v with one child of u.
function nniMove(root) {
    let internalNodes = [];
    function collect(n) {
        if (n.parent && n.childs && n.childs.length >= 2) internalNodes.push(n);
        for (let c of n.childs) collect(c);
    }
    collect(root);
    if (internalNodes.length === 0) return;

    let v = internalNodes[Math.floor(Math.random() * internalNodes.length)];
    let u = v.parent;
    if (!u || u.childs.length < 2) return;

    // Pick a sibling of v (subtree hanging off u that isn't v)
    let siblings = u.childs.filter(c => c !== v);
    if (siblings.length === 0) return;
    let w = siblings[Math.floor(Math.random() * siblings.length)];

    // Pick a random child of v
    let vChild = v.childs[Math.floor(Math.random() * v.childs.length)];

    // Swap: w enters v's children, vChild enters u's children
    let vIdx = v.childs.indexOf(vChild);
    let uIdx = u.childs.indexOf(w);
    v.childs[vIdx] = w;
    u.childs[uIdx] = vChild;
    w.parent = v;
    vChild.parent = u;
}

function mutateTree(tree) {
    let copy = cloneTree(tree);

    if (Math.random() < 0.6) {
        // Topology move (NNI) — changes which taxa are grouped together
        nniMove(copy);
    } else {
        // Branch length scaling move
        let nodes = [];
        function collect(n) {
            if (n.parent) nodes.push(n);
            for (let c of n.childs) collect(c);
        }
        collect(copy);
        if (nodes.length > 0) {
            for (let i = 0; i < 3; i++) {
                let n = nodes[Math.floor(Math.random() * nodes.length)];
                n.length *= Math.exp((Math.random() - 0.5) * 0.4);
                if (n.length < 0.001) n.length = 0.001;
            }
        }
    }
    return copy;
}

function runMCMCStep() {
    if(!state.mcmcRunning) return;

    for(let i=0; i<10; i++) { // chunk 10 sub-steps per animation frame
        state.mcmcStep++;

        // ── Block-update scheme (mimics BEAST operator schedule) ────────────
        // Each sub-step updates only ONE group of parameters so that each
        // likelihood term identifies its own parameters independently.
        // This prevents the non-identifiable drift of β ↔ μ through branch lengths.
        const roll = Math.random();

        if (roll < 0.33) {
            // Block A: μ only (identified by Felsenstein / sequence divergence)
            const propMu = state.currentMu * Math.exp((Math.random() - 0.5) * 0.5);
            const newLP  = computeLogPosterior(state.currentBirth, state.currentDeath, propMu, state.currentTree);
            if (Math.log(Math.random()) < newLP - state.currentLogPosterior) {
                state.currentMu = propMu;
                state.currentLogPosterior = newLP;
                state.accepted++;
            }

        } else if (roll < 0.66) {
            // Block B: β and δ only (identified by BD tree likelihood + root calibration)
            const propB = state.currentBirth * Math.exp((Math.random() - 0.5) * 0.5);
            const propD = state.currentDeath * Math.exp((Math.random() - 0.5) * 0.5);
            const newLP = computeLogPosterior(propB, propD, state.currentMu, state.currentTree);
            if (Math.log(Math.random()) < newLP - state.currentLogPosterior) {
                state.currentBirth = propB;
                state.currentDeath = propD;
                state.currentLogPosterior = newLP;
                state.accepted++;
            }

        } else {
            // Block C: tree topology and/or branch lengths
            const propTree = mutateTree(state.currentTree);
            const newLP    = computeLogPosterior(state.currentBirth, state.currentDeath, state.currentMu, propTree);
            if (Math.log(Math.random()) < newLP - state.currentLogPosterior) {
                state.currentTree = propTree;
                state.currentLogPosterior = newLP;
                state.accepted++;
            }
        }

        if(state.mcmcStep % 10 === 0) {
            state.traceData.gen.push(state.mcmcStep);
            state.traceData.birth.push(state.currentBirth);
            state.traceData.death.push(state.currentDeath);
            state.traceData.mu.push(state.currentMu);
            state.traceData.logPost.push(state.currentLogPosterior);
            
            if(state.traceData.gen.length > 10000) {
                state.traceData.gen.shift();
                state.traceData.birth.shift();
                state.traceData.death.shift();
                state.traceData.mu.shift();
                state.traceData.logPost.shift();
            }
        }
    }

    // Update UI
    let accRate = (state.accepted / state.mcmcStep * 100).toFixed(1);
    document.getElementById("stats-panel").innerHTML = `
        Generation: ${state.mcmcStep}<br>
        Acceptance: ${accRate}%<br>
        LogPosterior: ${state.currentLogPosterior.toFixed(2)}<br>
        Current β (Birth): ${state.currentBirth.toFixed(4)}<br>
        Current δ (Death): ${state.currentDeath.toFixed(4)}<br>
        Current μ (Mutation): ${state.currentMu.toFixed(5)}
    `;

    // Compute recovery statistics (post burn-in: last half of chain)
    const burnIn  = Math.floor(state.traceData.gen.length / 2);
    const bPost   = state.traceData.birth.slice(burnIn);
    const dPost   = state.traceData.death.slice(burnIn);
    const muPost  = state.traceData.mu.slice(burnIn);
    const nPost   = bPost.length;

    let bMean = '—', dMean = '—', muMean = '—';
    let bHPD = [NaN, NaN], dHPD = [NaN, NaN], muHPD = [NaN, NaN];
    let bESS = 0, dESS = 0, muESS = 0;

    if (nPost > 10) {
        bMean  = (bPost.reduce((a,b)=>a+b,0)  / nPost).toFixed(3);
        dMean  = (dPost.reduce((a,b)=>a+b,0)  / nPost).toFixed(3);
        muMean = (muPost.reduce((a,b)=>a+b,0) / nPost).toFixed(5);
        bHPD   = computeHPD(bPost);
        dHPD   = computeHPD(dPost);
        muHPD  = computeHPD(muPost);
        bESS   = computeESS(bPost);
        dESS   = computeESS(dPost);
        muESS  = computeESS(muPost);
    }

    const fmtHPD = (lo, hi, dp=3) => isNaN(lo) ? '—' : `[${lo.toFixed(dp)}, ${hi.toFixed(dp)}]`;

    if (state.isRealData) {
        document.getElementById("recovery-panel").innerHTML = `
            <table class="recovery-table">
                <tr><th>Param</th><th>Mean</th><th>95% HPD</th><th>ESS</th></tr>
                <tr>
                    <td>β</td>
                    <td style="color:blue;font-weight:bold;">${bMean}</td>
                    <td style="font-size:11px;">${fmtHPD(...bHPD)}</td>
                    <td>${bESS > 1 ? Math.round(bESS) : '—'}</td>
                </tr>
                <tr>
                    <td>δ</td>
                    <td style="color:red;font-weight:bold;">${dMean}</td>
                    <td style="font-size:11px;">${fmtHPD(...dHPD)}</td>
                    <td>${dESS > 1 ? Math.round(dESS) : '—'}</td>
                </tr>
                <tr>
                    <td>μ</td>
                    <td style="color:green;font-weight:bold;">${muMean}</td>
                    <td style="font-size:11px;">${fmtHPD(...muHPD, 5)}</td>
                    <td>${muESS > 1 ? Math.round(muESS) : '—'}</td>
                </tr>
            </table>
            <div style="font-size:11px;color:#777;margin-top:8px;">Real data · ½ burn-in discarded</div>
        `;
    } else {
        document.getElementById("recovery-panel").innerHTML = `
            <table class="recovery-table">
                <tr><th>Param</th><th>True</th><th>Mean</th><th>95% HPD</th><th>ESS</th></tr>
                <tr>
                    <td>β</td>
                    <td>${state.trueBirth.toFixed(3)}</td>
                    <td style="color:blue;font-weight:bold;">${bMean}</td>
                    <td style="font-size:11px;">${fmtHPD(...bHPD)}</td>
                    <td>${bESS > 1 ? Math.round(bESS) : '—'}</td>
                </tr>
                <tr>
                    <td>δ</td>
                    <td>${state.trueDeath.toFixed(3)}</td>
                    <td style="color:red;font-weight:bold;">${dMean}</td>
                    <td style="font-size:11px;">${fmtHPD(...dHPD)}</td>
                    <td>${dESS > 1 ? Math.round(dESS) : '—'}</td>
                </tr>
                <tr>
                    <td>μ</td>
                    <td>${state.trueMu.toFixed(5)}</td>
                    <td style="color:green;font-weight:bold;">${muMean}</td>
                    <td style="font-size:11px;">${fmtHPD(...muHPD, 5)}</td>
                    <td>${muESS > 1 ? Math.round(muESS) : '—'}</td>
                </tr>
            </table>
            <div style="font-size:11px;color:#777;margin-top:8px;">T, ρ, ψ, r fixed &middot; ½ burn-in discarded</div>
        `;
    }

    // Update trace chart
    chart.data.labels = state.traceData.gen;
    chart.data.datasets[0].data = state.traceData.birth;
    chart.data.datasets[1].data = state.traceData.death;
    // μ is plotted ×10 on the right axis so it's visible alongside β/δ
    chart.data.datasets[2].data = state.traceData.mu.map(v => v * 10);
    chart.update();

    // Update histograms every 100 steps (more expensive)
    if (state.mcmcStep % 100 === 0 && state.traceData.birth.length > 20) {
        updateHistograms();
    }

    // Store tree sample every 100 steps (keep last 300 for DensiTree)
    if (state.mcmcStep % 100 === 0) {
        state.treeSamples.push(cloneTree(state.currentTree));
        if (state.treeSamples.length > 300) state.treeSamples.shift();
    }

    // Refresh single current tree every 50 steps
    if (state.mcmcStep % 50 === 0) {
        drawTreeCanvas(state.currentTree, "tree-container");
        if (!state.isRealData && state.trueTree) {
            let dist = getTreeDistance(state.currentTree, state.trueTree);
            let distEl = document.getElementById("tree-dist");
            if (distEl) distEl.textContent = `RF dist: ${dist}`;
        }
    }

    // Redraw DensiTree every 200 steps
    if (state.mcmcStep % 200 === 0 && state.treeSamples.length > 0) {
        drawDensiTree();
    }

    requestAnimationFrame(runMCMCStep);
}

// Sort children at every internal node so the subtree containing the// lowest-numbered leaf (T1 < T2 < ...) comes first → leaves render top-to-bottom in order.
function sortTreeByLeafName(node) {
    if (!node.childs || node.childs.length === 0) return;
    for (let c of node.childs) sortTreeByLeafName(c);
    node.childs.sort((a, b) => minLeafNum(a) - minLeafNum(b));
}

function minLeafNum(node) {
    if (!node.childs || node.childs.length === 0) {
        // Extract numeric part from names like "T3" → 3
        let m = node.id && node.id.match(/\d+/);
        return m ? parseInt(m[0]) : Infinity;
    }
    return Math.min(...node.childs.map(minLeafNum));
}

function drawTreeCanvas(tree, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    // Measure container BEFORE touching canvas to avoid width inflation
    const W = container.offsetWidth || 600;
    const H = container.offsetHeight || 250;

    let canvas = container.querySelector("canvas");
    if (!canvas) {
        canvas = document.createElement("canvas");
        canvas.style.position = 'absolute';
        canvas.style.top = '0';
        canvas.style.left = '0';
        container.style.position = 'relative';
        container.style.overflow = 'hidden';
        container.innerHTML = "";
        container.appendChild(canvas);
    }

    canvas.width = W;
    canvas.height = H;

    // For the sampled tree, sort leaves so T1…TN appear top-to-bottom
    if (containerId === 'tree-container') sortTreeByLeafName(tree);

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, W, H);

    // --- Pass 1: compute maxDescDist for each node (max branch-length path to any tip)
    // Tips have maxDescDist = 0. Internal nodes inherit max child path + child.length.
    // x position = padX + (maxTreeDepth - node.maxDescDist) * scaleX
    // This guarantees all tips land at the same rightmost x.
    function computeMaxDesc(node) {
        if (!node.childs || node.childs.length === 0) {
            node._maxDesc = 0;
            return node.leaf_dead ? -Infinity : 0;  // dead nodes excluded from max
        }
        let best = -Infinity;
        for (let c of node.childs) {
            let d = (c.length || 0) + computeMaxDesc(c);
            if (d > best) best = d;
        }
        node._maxDesc = Math.max(best, 0);
        return node._maxDesc;
    }
    const treeDepth = computeMaxDesc(tree);  // = max root-to-tip distance

    // --- Pass 2: count live leaves for y spacing
    let leafCount = 0;
    function countLeaves(node) {
        if (!node.childs || node.childs.length === 0) {
            if (!node.leaf_dead) leafCount++;
            return;
        }
        for (let c of node.childs) countLeaves(c);
    }
    countLeaves(tree);

    if (treeDepth === 0 || leafCount === 0) return;

    const padL = 10;          // left pad
    const padR = 60;          // right pad (room for tip labels)
    const padY = 10;
    const drawW = W - padL - padR;
    const drawH = H - padY * 2;
    const scaleX = drawW / treeDepth;
    const scaleY = drawH / leafCount;

    let currentLeaf = 0;

    ctx.strokeStyle = "#333";
    ctx.lineWidth = 1.5;
    ctx.font = "12px monospace";
    ctx.fillStyle = "#333";

    // Returns the canvas y of this node
    function drawNode(node) {
        // x is determined by how far we are from the tips
        const x = padL + (treeDepth - node._maxDesc) * scaleX;

        if (!node.childs || node.childs.length === 0) {
            // Skip extinct lineages — they kept their internal NX id and should not render
            if (node.leaf_dead) return null;
            // Tip — all tips align at x = padL + treeDepth * scaleX
            const y = padY + currentLeaf * scaleY + scaleY / 2;
            currentLeaf++;
            // Only label named extant tips (T1, T2, …) — skip internal node IDs
            if (node.id && node.id.startsWith('T')) {
                ctx.fillText(node.id, x + 4, y + 4);
            }
            node._canvasY = y;
            return y;
        }

        let childYs = [];
        for (let c of node.childs) {
            const cy = drawNode(c);
            if (cy === null) continue;   // dead branch — skip
            const cx = padL + (treeDepth - c._maxDesc) * scaleX;

            // Horizontal branch from this node's x to child's x
            ctx.beginPath();
            ctx.moveTo(x, cy);
            ctx.lineTo(cx, cy);
            ctx.stroke();

            childYs.push(cy);
        }

        // Vertical connector between topmost and bottommost child
        if (childYs.length === 0) return null;  // all children extinct — nothing to draw
        const topY = Math.min(...childYs);
        const botY = Math.max(...childYs);
        ctx.beginPath();
        ctx.moveTo(x, topY);
        ctx.lineTo(x, botY);
        ctx.stroke();

        const y = (topY + botY) / 2;
        node._canvasY = y;
        return y;
    }

    drawNode(tree);
}

// DensiTree: overlay all sampled trees at low opacity.
// Leaves are fixed to the same y positions across all trees so
// uncertainty shows as spread in the x/branching dimension.
function drawDensiTree() {
    const container = document.getElementById('densi-container');
    if (!container) return;
    const samples = state.treeSamples;

    const W = container.offsetWidth || 600;
    const H = container.offsetHeight || 500;

    let canvas = container.querySelector('canvas');
    if (!canvas) {
        canvas = document.createElement('canvas');
        canvas.style.position = 'absolute';
        canvas.style.top = '0';
        canvas.style.left = '0';
        container.style.position = 'relative';
        container.style.overflow = 'hidden';
        container.innerHTML = '';
        container.appendChild(canvas);
    }
    canvas.width = W;
    canvas.height = H;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, W, H);
    if (samples.length === 0) return;

    // Build fixed global leaf order — from true tree if available, else from first sample.
    const leafOrder = [];
    function collectLeafNames(node) {
        if (!node.childs || node.childs.length === 0) {
            if (!node.leaf_dead && node.id) leafOrder.push(node.id);
            return;
        }
        for (const c of node.childs) collectLeafNames(c);
    }
    const refTree = state.trueTree || samples[0];
    collectLeafNames(refTree);
    // Sort: numeric suffix first (T1, T2…), then alphabetical for real data labels
    leafOrder.sort((a, b) => {
        const na = parseInt(a.replace(/\D/g, ''));
        const nb = parseInt(b.replace(/\D/g, ''));
        if (!isNaN(na) && !isNaN(nb)) return na - nb;
        return a.localeCompare(b);
    });

    const nLeaves = leafOrder.length;
    if (nLeaves === 0) return;

    const padL = 10, padR = 60, padY = 15;
    const drawW = W - padL - padR;
    const drawH = H - padY * 2;
    const scaleY = drawH / nLeaves;

    // Fixed y for each leaf label
    const leafY = {};
    leafOrder.forEach((name, i) => {
        leafY[name] = padY + i * scaleY + scaleY / 2;
    });

    // Compute maxDesc per node so tips align at same x (ultrametric display)
    function computeMaxDescLocal(node) {
        if (!node.childs || node.childs.length === 0) {
            node._dd = node.leaf_dead ? -Infinity : 0;
            return node._dd;
        }
        let best = -Infinity;
        for (const c of node.childs) {
            const d = (c.length || 0) + computeMaxDescLocal(c);
            if (d > best) best = d;
        }
        node._dd = Math.max(best, 0);
        return node._dd;
    }

    // Find global max depth for a stable shared x scale
    let globalMaxDepth = 0;
    for (const t of samples) {
        const d = computeMaxDescLocal(t);
        if (d > globalMaxDepth) globalMaxDepth = d;
    }
    if (globalMaxDepth === 0) globalMaxDepth = 1;
    const scaleX = drawW / globalMaxDepth;

    // Draw each tree semi-transparently
    // Alpha decreases as sample count grows so overlay stays readable
    const alpha = Math.max(0.04, Math.min(0.2, 6 / samples.length));
    ctx.lineWidth = 1;

    function drawSample(node) {
        const nx = padL + (globalMaxDepth - (node._dd || 0)) * scaleX;

        if (!node.childs || node.childs.length === 0) {
            if (node.leaf_dead) return null;
            const name = node.id && node.id.startsWith('T') ? node.id : null;
            return name ? (leafY[name] ?? null) : null;
        }

        const childYs = [];
        for (const c of node.childs) {
            const cy = drawSample(c);
            if (cy === null) continue;
            const cx = padL + (globalMaxDepth - (c._dd || 0)) * scaleX;
            ctx.beginPath();
            ctx.moveTo(nx, cy);
            ctx.lineTo(cx, cy);
            ctx.stroke();
            childYs.push(cy);
        }

        if (childYs.length === 0) return null;

        const topY = Math.min(...childYs);
        const botY = Math.max(...childYs);
        ctx.beginPath();
        ctx.moveTo(nx, topY);
        ctx.lineTo(nx, botY);
        ctx.stroke();

        return (topY + botY) / 2;
    }

    ctx.strokeStyle = `rgba(0,80,180,${alpha})`;
    for (const t of samples) {
        sortTreeByLeafName(t);
        computeMaxDescLocal(t);
        drawSample(t);
    }

    // Draw tip labels once on top
    ctx.fillStyle = '#333';
    ctx.font = '12px monospace';
    const tipX = padL + globalMaxDepth * scaleX;
    leafOrder.forEach(name => {
        ctx.fillText(name, tipX + 4, leafY[name] + 4);
    });

    const el = document.getElementById('densi-count');
    if (el) el.textContent = `${samples.length} trees`;
}
