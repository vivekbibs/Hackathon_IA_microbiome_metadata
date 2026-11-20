
// =====================================================
// Configuration
// =====================================================

const API_URL = "";

let currentPMCID = null;
let isLoading = false;


// =====================================================
// DOM elements
// =====================================================

const pmcidInput = document.getElementById("pmcid");
const loadArticleBtn = document.getElementById("loadArticleBtn");
const articleStatus = document.getElementById("articleStatus");

const articleTitle = document.getElementById("articleTitle");
const articleJournal = document.getElementById("articleJournal");
const articleDate = document.getElementById("articleDate");
const articleAuthors = document.getElementById("articleAuthors");
const articleChunks = document.getElementById("articleChunks");

const welcomeScreen = document.getElementById("welcomeScreen");
const messagesContainer = document.getElementById("messages");

const questionInput = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");


// =====================================================
// Utility functions
// =====================================================

function setStatus(message, type = "") {
    articleStatus.textContent = message;
    articleStatus.className = "status-message " + type;
}


function setChatEnabled(enabled) {
    questionInput.disabled = !enabled;
    sendBtn.disabled = !enabled;

    questionInput.placeholder = enabled
        ? "Ask a question about this article..."
        : "Load an article first...";
}


function getCurrentTime() {
    return new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit"
    });
}


function showError(message) {
    alert(message);
}


// =====================================================
// Load article
// =====================================================

async function loadArticle() {

    if (isLoading) return;

    const pmcid = pmcidInput.value.trim().toUpperCase();

    if (!/^PMC\d+$/.test(pmcid)) {
        setStatus("Please enter a valid PMCID (e.g. PMC1234567).", "error");
        return;
    }

    isLoading = true;

    loadArticleBtn.disabled = true;
    loadArticleBtn.textContent = "Loading article...";

    setStatus("Downloading article and preparing embeddings...");

    try {

        const response = await fetch(
            `${API_URL}/articles/${encodeURIComponent(pmcid)}/load`,
            {
                method: "POST"
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Could not load article.");
        }

        currentPMCID = pmcid;

        displayArticleMetadata(data);

        setStatus("Article loaded successfully.", "success");

        welcomeScreen.style.display = "block";
        messagesContainer.innerHTML = "";

        setChatEnabled(true);

    } catch (error) {

        currentPMCID = null;
        setChatEnabled(false);

        setStatus(error.message, "error");

    } finally {

        isLoading = false;

        loadArticleBtn.disabled = false;
        loadArticleBtn.textContent = "Load article";

    }
}


// =====================================================
// Display metadata
// =====================================================

function displayArticleMetadata(data) {

    const metadata = data.metadata || {};

    articleTitle.textContent =
        metadata.title || "Title not available";

    articleJournal.textContent =
        metadata.journal || "-";

    articleDate.textContent =
        metadata.date || "-";

    articleAuthors.textContent =
        Array.isArray(metadata.authors)
            ? metadata.authors.join(", ")
            : metadata.authors || "-";

    articleChunks.textContent =
        data.chunks ?? "-";
}


// =====================================================
// Add message to chat
// =====================================================

function addMessage(text, sender) {

    welcomeScreen.style.display = "none";

    const message = document.createElement("div");
    message.className = `message ${sender}`;

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = sender === "user" ? "👤" : "🧬";

    const content = document.createElement("div");
    content.className = "message-content";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";

    // textContent prevents HTML injection
    bubble.textContent = text;

    const time = document.createElement("div");
    time.className = "message-time";
    time.textContent = getCurrentTime();

    content.appendChild(bubble);
    content.appendChild(time);

    message.appendChild(avatar);
    message.appendChild(content);

    messagesContainer.appendChild(message);

    scrollToBottom();
}


// =====================================================
// Typing indicator
// =====================================================

function showTypingIndicator() {

    const message = document.createElement("div");
    message.className = "message";
    message.id = "typingIndicator";

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = "🧬";

    const typing = document.createElement("div");
    typing.className = "typing";

    for (let i = 0; i < 3; i++) {
        const dot = document.createElement("span");
        typing.appendChild(dot);
    }

    message.appendChild(avatar);
    message.appendChild(typing);

    messagesContainer.appendChild(message);

    scrollToBottom();
}


function removeTypingIndicator() {

    const indicator = document.getElementById("typingIndicator");

    if (indicator) {
        indicator.remove();
    }
}


// =====================================================
// Scroll chat
// =====================================================

function scrollToBottom() {

    const chatContainer = document.querySelector(".chat-container");

    chatContainer.scrollTo({
        top: chatContainer.scrollHeight,
        behavior: "smooth"
    });
}


// =====================================================
// Send question
// =====================================================

async function sendQuestion() {

    if (isLoading || !currentPMCID) return;

    const question = questionInput.value.trim();

    if (!question) return;

    isLoading = true;

    questionInput.value = "";
    sendBtn.disabled = true;

    addMessage(question, "user");

    showTypingIndicator();

    try {

        const response = await fetch(`${API_URL}/query`, {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                pmcid: currentPMCID,
                question: question
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Error while querying the article.");
        }

        removeTypingIndicator();

        addMessage(data.answer, "assistant");

    } catch (error) {

        removeTypingIndicator();

        addMessage(
            `Sorry, an error occurred: ${error.message}`,
            "assistant"
        );

    } finally {

        isLoading = false;
        sendBtn.disabled = false;
        questionInput.focus();

    }
}


// =====================================================
// Suggested questions
// =====================================================

document.querySelectorAll(".suggestion-btn").forEach(button => {

    button.addEventListener("click", () => {

        if (!currentPMCID) {
            setStatus("Please load an article first.", "error");
            return;
        }

        questionInput.value = button.dataset.question;

        sendQuestion();

    });

});


// =====================================================
// Event listeners
// =====================================================

loadArticleBtn.addEventListener("click", loadArticle);

sendBtn.addEventListener("click", sendQuestion);


questionInput.addEventListener("keydown", event => {

    if (event.key === "Enter" && !event.shiftKey) {

        event.preventDefault();

        sendQuestion();

    }

});


pmcidInput.addEventListener("keydown", event => {

    if (event.key === "Enter") {
        loadArticle();
    }

});


questionInput.addEventListener("input", () => {

    questionInput.style.height = "auto";

    questionInput.style.height =
        Math.min(questionInput.scrollHeight, 120) + "px";

});


// =====================================================
// Initialization
// =====================================================

setChatEnabled(false);
