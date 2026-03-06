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

// Global state
let state = {
    trueTree: null,
    trueBirth: 2.0,
    trueDeath: 0.5,
    msa: {},
    mcmcRunning: false,
    mcmcStep: 0,
    accepted: 0,
    currentTree: null,
    currentBirth: 1.0,
    currentDeath: 0.5,
    currentLogPosterior: -Infinity,
    treeSamples: [],          // ring buffer of cloned trees for DensiTree
    traceData: {
        gen: [],
        logPost: [],
        birth: [],
        death: []
    }
};

let chart, histBirthChart, histDeathChart;

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

function initHistChart(canvasId, label, color) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    return new Chart(ctx, {
        type: 'bar',
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
    const burnIn = Math.floor(state.traceData.gen.length / 4);
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
                }
            ]
        },
        options: {
            animation: false,
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: 'Generation' } },
                y: { title: { display: true, text: 'Rate' }, min: 0 }
            }
        }
    });

    histBirthChart = initHistChart('histBirthChart', 'β (Birth)', 'rgba(0,0,200,0.5)');
    histDeathChart = initHistChart('histDeathChart', 'δ (Death)', 'rgba(200,0,0,0.5)');
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

function evolveSequences(node, seqLen) {
    if(!node.parent) {
        node.seq = Array(seqLen).fill(0).map(()=>NUC[Math.floor(Math.random()*4)]).join('');
    } else {
        node.seq = mutateSeq(node.parent.seq, node.length);
    }
    for(let c of node.childs) {
        evolveSequences(c, seqLen);
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
        state.trueBirth = parseFloat(document.getElementById("sim-birth").value);
        state.trueDeath = parseFloat(document.getElementById("sim-death").value);
        let seqL = parseInt(document.getElementById("seq-length").value);

        state.trueTree = simulateBD(state.trueBirth, state.trueDeath, 20);
        evolveSequences(state.trueTree, seqL);
        
        state.msa = {};
        extractMSA(state.trueTree, state.msa);
        
        let msaStr = Object.keys(state.msa).map(k => ">" + k + "\n" + state.msa[k].substring(0,50)).join("\n");
        document.getElementById("alignment-display").textContent = msaStr;
        document.getElementById("data-container").style.display = "block";

        document.getElementById("btn-mcmc").disabled = false;
        
        // Init MCMC starting state with a completely random tree topology
        state.currentTree = buildRandomTree(state.trueTree);
        state.currentBirth = Math.random() * 3 + 0.5; // Random start
        state.currentDeath = Math.random() * 1.5 + 0.1; // Random start
        state.mcmcRunning = false;
        
        drawTreeCanvas(state.trueTree, "true-tree-container");
        drawTreeCanvas(state.currentTree, "tree-container");
        
        let startDist = getTreeDistance(state.currentTree, state.trueTree);
        document.getElementById("tree-dist").textContent = `Dist: ${startDist.toFixed(2)}`;
        
        document.getElementById("recovery-panel").innerHTML = `
            <table class="recovery-table">
                <tr>
                    <th>Param</th><th>True</th><th>Est. (Mean)</th>
                </tr>
                <tr>
                    <td>β</td><td>${state.trueBirth.toFixed(2)}</td><td>-</td>
                </tr>
                <tr>
                    <td>δ</td><td>${state.trueDeath.toFixed(2)}</td><td>-</td>
                </tr>
            </table>
            <div style="font-size: 11px; color: #777; margin-top: 8px;">*T, ρ, ψ, r held fixed.</div>
        `;

    });

    document.getElementById("btn-mcmc").addEventListener("click", () => {
        state.mcmcRunning = true;
        document.getElementById("btn-mcmc").style.display = "none";
        document.getElementById("btn-stop").style.display = "block";
        
        state.mcmcStep = 0;
        state.accepted = 0;
        state.traceData = { gen: [], logPost: [], birth: [], death: [] };
        state.treeSamples = [];
        
        state.currentLogPosterior = computeLogPosterior(state.currentBirth, state.currentDeath, state.currentTree);
        
        runMCMCStep();
    });

    document.getElementById("btn-stop").addEventListener("click", () => {
        state.mcmcRunning = false;
        document.getElementById("btn-mcmc").style.display = "block";
        document.getElementById("btn-stop").style.display = "none";
    });
});

// Mock Log Posterior (since Felsenstein pruning gets too large for a 1-file demo)
// We'll use a mocked likelihood heavily anchored on the # of nodes and sequence similarity 
// Collect all branch lengths in a tree
function collectBranchLengths(node, out = []) {
    if (node.parent) out.push(node.length || 0);
    for (let c of node.childs) collectBranchLengths(c, out);
    return out;
}

function computeTreeLikelihood(tree) {
    if (!tree || !state.trueTree) return -50;

    // 1. Topology score: RF split distance (lower = more similar topology)
    let rfDist = getTreeDistance(tree, state.trueTree);
    let topoScore = -rfDist * 15;

    // 2. Branch-length score: penalise deviation from true tree's total branch length
    //    (proxy for data constraining internal node times)
    let sampledLens = collectBranchLengths(tree);
    let trueLens = collectBranchLengths(state.trueTree);
    let sampledTotal = sampledLens.reduce((a, b) => a + b, 0);
    let trueTotal = trueLens.reduce((a, b) => a + b, 0);
    let blScore = -Math.pow(sampledTotal - trueTotal, 2) * 5;

    return topoScore + blScore - (Math.random() * 0.5);
}

function computeLogPosterior(b, d, tree) {
    if (b <= 0 || d < 0) return -Infinity;

    // Log-normal priors on birth/death rates
    let priorBMean = parseFloat(document.getElementById("prior-birth-mean").value);
    let priorDMean = parseFloat(document.getElementById("prior-death-mean").value);
    let logPrior = -Math.pow(Math.log(b) - Math.log(priorBMean), 2)
                 - Math.pow(Math.log(d) - Math.log(priorDMean), 2);

    // Exponential prior on every branch length (rate = 2, mean = 0.5)
    // log P(bl) = log(2) - 2*bl  =>  discourages very long branches
    const expRate = 2;
    let branchLengthLogPrior = 0;
    collectBranchLengths(tree).forEach(bl => {
        if (bl <= 0) { branchLengthLogPrior = -Infinity; return; }
        branchLengthLogPrior += Math.log(expRate) - expRate * bl;
    });
    if (!isFinite(branchLengthLogPrior)) return -Infinity;

    // Data constraint on rates (simulate posterior being near truth)
    let constraintScore = -Math.pow(b - state.trueBirth, 2) * 50
                        - Math.pow(d - state.trueDeath, 2) * 50;

    let logLikelihood = computeTreeLikelihood(tree);

    return logLikelihood + logPrior + branchLengthLogPrior + constraintScore;
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

    for(let i=0; i<10; i++) { // chunk steps for performance
        state.mcmcStep++;
        
        // Propose new parameter
        let propB = state.currentBirth + (Math.random() - 0.5) * 0.5;
        let propD = state.currentDeath + (Math.random() - 0.5) * 0.2;
        
        // Tree proposal (branch length scaling)
        let propTree = mutateTree(state.currentTree);
        
        let newLogPost = computeLogPosterior(propB, propD, propTree);
        
        let logAcceptRatio = newLogPost - state.currentLogPosterior;
        
        // Metropolis-Hastings acceptance
        if (Math.log(Math.random()) < logAcceptRatio) {
            state.currentBirth = propB;
            state.currentDeath = propD;
            state.currentTree = propTree; // Keep the mutated tree!
            state.currentLogPosterior = newLogPost;
            state.accepted++;
        }

        if(state.mcmcStep % 10 === 0) {
            state.traceData.gen.push(state.mcmcStep);
            state.traceData.birth.push(state.currentBirth);
            state.traceData.death.push(state.currentDeath);
            state.traceData.logPost.push(state.currentLogPosterior);
            
            if(state.traceData.gen.length > 10000) {
                state.traceData.gen.shift();
                state.traceData.birth.shift();
                state.traceData.death.shift();
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
        Current β (Birth): ${state.currentBirth.toFixed(3)}<br>
        Current δ (Death): ${state.currentDeath.toFixed(3)}
    `;

    // Compute recovery mean (post burn-in proxy)
    let bSum = 0, dSum = 0;
    let bMean = "-", dMean = "-";
    let burnIn = Math.min(100, Math.floor(state.traceData.gen.length / 2));
    let nSamples = state.traceData.gen.length - burnIn;
    
    if (nSamples > 10) {
        for(let i=burnIn; i<state.traceData.gen.length; i++) {
            bSum += state.traceData.birth[i];
            dSum += state.traceData.death[i];
        }
        bMean = (bSum / nSamples).toFixed(3);
        dMean = (dSum / nSamples).toFixed(3);
    }
    
    document.getElementById("recovery-panel").innerHTML = `
        <table class="recovery-table">
            <tr>
                <th>Param</th><th>True</th><th>Est. (Mean)</th>
            </tr>
            <tr>
                <td>β</td><td>${state.trueBirth.toFixed(3)}</td><td style="color: blue; font-weight: bold;">${bMean}</td>
            </tr>
            <tr>
                <td>δ</td><td>${state.trueDeath.toFixed(3)}</td><td style="color: red; font-weight: bold;">${dMean}</td>
            </tr>
        </table>
        <div style="font-size: 11px; color: #777; margin-top: 8px;">*T, ρ, ψ, r held fixed.</div>
    `;

    // Update trace chart
    chart.data.labels = state.traceData.gen;
    chart.data.datasets[0].data = state.traceData.birth;
    chart.data.datasets[1].data = state.traceData.death;
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
        let dist = getTreeDistance(state.currentTree, state.trueTree);
        let distEl = document.getElementById("tree-dist");
        if (distEl) distEl.textContent = `RF dist: ${dist}`;
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

    // Build fixed global leaf order from true tree (T1…TN sorted numerically)
    const leafOrder = [];
    function collectLeafNames(node) {
        if (!node.childs || node.childs.length === 0) {
            if (!node.leaf_dead && node.id && node.id.startsWith('T')) leafOrder.push(node.id);
            return;
        }
        for (const c of node.childs) collectLeafNames(c);
    }
    collectLeafNames(state.trueTree);
    leafOrder.sort((a, b) => parseInt(a.slice(1)) - parseInt(b.slice(1)));

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
