/*******************************************************
 PARSON FRONTEND CONTROLLER
 Handles:
 - Button click
 - API communication
 - Rendering results
 - Visual explainability bars
*******************************************************/


/* ============================
   DOM ELEMENT REFERENCES
============================ */

const queryInput = document.getElementById("queryInput");
const searchBtn = document.getElementById("searchBtn");
const resultsDiv = document.getElementById("results");
const resultsHeader = document.getElementById("results-header");
const loadingDiv = document.getElementById("loading");
const errorDiv = document.getElementById("error");
const historyList = document.getElementById("historyList");
queryInput.addEventListener("keydown", function(event) {

    if (event.key === "Enter" && !event.shiftKey) {

        event.preventDefault();

        searchBtn.click();
    }
});


/* ============================
   API ENDPOINT
============================ */

const API_URL = "https://parson-production.up.railway.app/recommend";


/* ============================
   SEARCH BUTTON CLICK
============================ */

searchBtn.addEventListener("click", async () => {

    const query = queryInput.value.trim();

    if (!query) {
        alert("Please enter a query");
        return;
    }

    addToHistory(query);
    fetchRecommendations(query);
});


/* ============================
   CALL BACKEND API
============================ */

async function fetchRecommendations(query) {

    resultsDiv.innerHTML = "";
    errorDiv.innerText = "";
    loadingDiv.innerHTML = `
        <div class="loading-animation">

            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>

            <span>
                PARSON is analyzing themes and matching books...
            </span>

        </div>
    `;

    try {

        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                query: query,
                top_k: 10
            })
        });

        const data = await response.json();
        loadingDiv.innerText = "";

        displayResults(data.results, query);

    } catch (error) {
        loadingDiv.innerText = "";
        errorDiv.innerText = "Failed to connect to server.";
    }
}


/* ============================
   DISPLAY RESULTS
============================ */

function displayResults(books, query) {

    if (!books || books.length === 0) {
        resultsDiv.innerHTML = "<p>No results found.</p>";
        return;
    }
    resultsHeader.innerHTML = `

        <div class="results-ai-header">

            <div class="results-query">
                Top matches for:
                <span>"${query}"</span>
            </div>

            <div class="results-themes">
                PARSON detected themes:
                ${detectThemes(query)}
            </div>

        </div>
    `;

    books.forEach((book, index) => {

        const card = document.createElement("div");
        card.className = "card";
        card.style.animationDelay = `${index * 0.12}s`;
        card.innerHTML = `
    <div class="card-layout">

        ${book.thumbnail ? `
            <div class="thumbnail-section">
                <img
                    src="${book.thumbnail}"
                    class="thumbnail"
                    alt="Book Cover"
                >
            </div>
        ` : ""}

        <div class="content-section">

            <h2 class="book-title">
                ${book.title}
            </h2>

            ${book.author ? `
                <p class="book-author">
                    ${book.author}
                </p>
            ` : ""}
            <div class="section-title">
                Synopsis
            </div>
            <p class="summary">
                ${book.summary || "No summary available."}
            </p>

            <div class="tag-container">
                ${generateTags(book, query)}
            </div>

            <div class="section-title">
                Why PARSON Recommended This :
            </div>
            <p class="reason">
                ${book.reason}
            </p>

            <a href="${book.preview_link}" target="_blank" class="preview-btn">
                Preview Book
            </a>

        </div>

    </div>
`;
        resultsDiv.appendChild(card);
    });
}


/* ============================
   BAR GENERATOR
============================ */

function createBar(label, value) {

    const percent = Math.min(Math.max(value * 100, 0), 100).toFixed(0);

    return `
        <div class="bar-block">
            <span>${label}: ${percent}%</span>
            <div class="bar-bg">
                <div class="bar-fill" style="width:${percent}%"></div>
            </div>
        </div>
    `;
}


/* ============================
   NORMALIZE INTENT VALUE
============================ */

function normalizeIntent(v) {
    return (v + 1) / 2;   // convert -1..1 → 0..1
}


/* ============================
   SEARCH HISTORY
============================ */

function addToHistory(query) {

    const item = document.createElement("div");
    item.className = "history-item";
    item.innerText = query;
    item.addEventListener("click", () => {

    queryInput.value = query;

    searchBtn.click();
});
    historyList.prepend(item);
}
document.querySelectorAll(".suggestion-chip")
.forEach(chip => {

    chip.addEventListener("click", () => {

        queryInput.value = chip.innerText;

        searchBtn.click();
    });
});
function detectThemes(query) {

    const q = query.toLowerCase();

    const themes = [];

    if (q.includes("ai") || q.includes("artificial")) {
        themes.push("Artificial Intelligence");
    }

    if (q.includes("space")) {
        themes.push("Space Exploration");
    }

    if (q.includes("alien")) {
        themes.push("Alien Civilizations");
    }

    if (q.includes("wizard") || q.includes("magic")) {
        themes.push("Magic");
    }

    if (q.includes("fantasy")) {
        themes.push("Fantasy");
    }

    if (q.includes("crime")) {
        themes.push("Crime");
    }

    if (q.includes("future")) {
        themes.push("Future Technology");
    }

    if (q.includes("war")) {
        themes.push("Conflict");
    }

    if (themes.length === 0) {
        themes.push("Story Discovery");
    }

    return themes.join(" • ");
}
/* ============================
   DYNAMIC TAG GENERATOR
============================ */

function generateTags(book, query) {

    const text = `
        ${book.title || ""}
        ${book.summary || ""}
        ${query || ""}
    `.toLowerCase();

    const tags = [];

    // AI / Technology
    if (
        text.includes("ai") ||
        text.includes("robot") ||
        text.includes("technology") ||
        text.includes("artificial")
    ) {
        tags.push("AI");
    }

    // Fantasy
    if (
        text.includes("magic") ||
        text.includes("wizard") ||
        text.includes("dragon") ||
        text.includes("fantasy")
    ) {
        tags.push("Fantasy");
    }

    // Space
    if (
        text.includes("space") ||
        text.includes("galaxy") ||
        text.includes("alien")
    ) {
        tags.push("Sci-Fi");
    }

    // Crime
    if (
        text.includes("crime") ||
        text.includes("murder") ||
        text.includes("detective")
    ) {
        tags.push("Crime");
    }

    // War
    if (
        text.includes("war") ||
        text.includes("battle")
    ) {
        tags.push("Conflict");
    }

    // Emotional
    if (
        text.includes("healing") ||
        text.includes("emotion") ||
        text.includes("lonely")
    ) {
        tags.push("Emotional");
    }

    // Adventure
    if (
        text.includes("adventure") ||
        text.includes("journey")
    ) {
        tags.push("Adventure");
    }

    // Fallback
    if (tags.length === 0) {
        tags.push("Recommended");
    }

    // Limit tags
    return tags.slice(0, 4).map(tag => `
        <span class="tag">
            ${tag}
        </span>
    `).join("");
}
/* ============================
   RECOMMENDATION BADGES
============================ */

function generateRecommendationBadge(index) {

    const badges = [

        "Excellent Match",

        "Best Semantic Match",

        "Highly Recommended",

        "Reader Favorite",

        "Hidden Gem"
    ];

    return badges[index % badges.length];
}
