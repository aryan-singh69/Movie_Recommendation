(function () {
    const API_BASE = "/api";
    const TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p";
    const PLACEHOLDER_POSTER = "https://placehold.co/400x600/161616/ffffff?text=No+Poster";

    const storage = {
        token: "access_token",
        userId: "user_id",
        username: "username",
        pendingSearch: "pending_search"
    };

    function $(selector, root = document) {
        return root.querySelector(selector);
    }

    function $all(selector, root = document) {
        return Array.from(root.querySelectorAll(selector));
    }

    function token() {
        return localStorage.getItem(storage.token);
    }

    function userId() {
        return localStorage.getItem(storage.userId);
    }

    function username() {
        return localStorage.getItem(storage.username);
    }

    function setSession(data) {
        localStorage.setItem(storage.token, data.access_token);
        localStorage.setItem(storage.userId, data.user_id);
        localStorage.setItem(storage.username, data.username);
        updateAuthUi();
    }

    function clearSession() {
        localStorage.removeItem(storage.token);
        localStorage.removeItem(storage.userId);
        localStorage.removeItem(storage.username);
        updateAuthUi();
    }

    function updateAuthUi() {
        const isAuthenticated = Boolean(token());
        document.body.classList.toggle("is-authenticated", isAuthenticated);

        const name = username() || "User";
        $all(".username-display").forEach((el) => {
            el.textContent = name;
        });
        $all(".user-avatar").forEach((el) => {
            el.textContent = name.trim().charAt(0).toUpperCase() || "U";
        });
    }

    function showToast(message, type = "info") {
        const host = $("#toastHost");
        if (!host) return;

        const toast = document.createElement("div");
        toast.className = `toast ${type}`;
        toast.textContent = message;
        host.appendChild(toast);

        window.setTimeout(() => {
            toast.style.opacity = "0";
            toast.style.transform = "translateY(8px)";
            window.setTimeout(() => toast.remove(), 180);
        }, 3600);
    }

    function setState(elementOrId, message, type = "info") {
        const el = typeof elementOrId === "string" ? document.getElementById(elementOrId) : elementOrId;
        if (!el) return;

        el.textContent = message || "";
        el.className = `state-message ${type === "error" ? "error" : type === "success" ? "success" : ""}`;
        el.classList.toggle("hidden", !message);
    }

    async function parseResponse(response) {
        const contentType = response.headers.get("content-type") || "";
        if (contentType.includes("application/json")) {
            return response.json();
        }
        return { detail: await response.text() };
    }

    async function apiFetch(url, options = {}) {
        const response = await fetch(url, {
            ...options,
            headers: {
                "Content-Type": "application/json",
                ...(options.headers || {})
            }
        });
        const data = await parseResponse(response);
        if (!response.ok) {
            throw new Error(data.detail || `Request failed with status ${response.status}`);
        }
        return data;
    }

    async function authFetch(url, options = {}) {
        const jwt = token();
        if (!jwt) {
            redirectToLogin();
            return null;
        }

        try {
            return await apiFetch(url, {
                ...options,
                headers: {
                    Authorization: `Bearer ${jwt}`,
                    ...(options.headers || {})
                }
            });
        } catch (error) {
            if (String(error.message).toLowerCase().includes("credentials") || String(error.message).includes("401")) {
                logout(false);
                return null;
            }
            throw error;
        }
    }

    function redirectToLogin() {
        if (!["/login", "/signup", "/"].includes(window.location.pathname)) {
            window.location.href = "/login";
        }
    }

    function posterUrl(path, size = "w342") {
        if (!path) return PLACEHOLDER_POSTER;
        if (/^https?:\/\//i.test(path)) return path;
        return `${TMDB_IMAGE_BASE}/${size}${path}`;
    }

    function normalizeGenres(genres) {
        if (!genres) return [];
        return String(genres)
            .split(/[|,]/)
            .map((genre) => genre.trim())
            .filter(Boolean)
            .slice(0, 4);
    }

    function movieTitle(movie) {
        return movie.title || movie.title_ml || movie.title_tmdb || "Untitled";
    }

    function movieRating(movie) {
        const rating = movie.global_avg_rating ?? movie.vote_average ?? movie.rating ?? 0;
        const value = Number(rating);
        return Number.isFinite(value) ? value.toFixed(1) : "0.0";
    }

    function createMovieCard(movie) {
        const card = document.createElement("article");
        card.className = "movie-card";
        card.tabIndex = 0;
        card.setAttribute("role", "button");
        card.setAttribute("aria-label", `Open details for ${movieTitle(movie)}`);

        const id = movie.movieId ?? movie.id;
        const genres = normalizeGenres(movie.genres).join(", ") || "Movie";
        const title = movieTitle(movie);

        card.innerHTML = `
            <img class="movie-poster" src="${posterUrl(movie.poster_path)}" alt="${escapeHtml(title)} poster" loading="lazy">
            <div class="movie-card-body">
                <h3 class="movie-title">${escapeHtml(title)}</h3>
                <p class="movie-genres">${escapeHtml(genres)}</p>
                <div class="movie-meta">
                    <span>${movieRating(movie)} / 5</span>
                    <span>Details</span>
                </div>
            </div>
        `;

        const open = () => {
            if (id !== undefined && id !== null) {
                window.location.href = `/movie/${id}`;
            }
        };
        card.addEventListener("click", open);
        card.addEventListener("keydown", (event) => {
            if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                open();
            }
        });
        return card;
    }

    function escapeHtml(value) {
        return String(value)
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#039;");
    }

    function renderMovieGrid(containerId, movies, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) return;

        container.innerHTML = "";
        const list = Array.isArray(movies) ? movies : [];
        if (!list.length) {
            if (options.emptyStateId) {
                setState(options.emptyStateId, options.emptyMessage || "No movies found.");
            } else {
                container.innerHTML = `<div class="state-message">${escapeHtml(options.emptyMessage || "No movies found.")}</div>`;
            }
            return;
        }

        if (options.emptyStateId) setState(options.emptyStateId, "");
        list.forEach((movie) => container.appendChild(createMovieCard(movie)));
    }

    function renderSkeleton(containerId, count = 6) {
        const container = document.getElementById(containerId);
        if (!container) return;
        container.innerHTML = "";
        for (let i = 0; i < count; i += 1) {
            const skeleton = document.createElement("div");
            skeleton.className = "skeleton-card";
            container.appendChild(skeleton);
        }
    }

    async function signup(usernameValue, email, password) {
        const data = await apiFetch("/auth/signup", {
            method: "POST",
            body: JSON.stringify({ username: usernameValue, email, password })
        });
        setSession(data);
        showToast("Account created. Welcome to MovieRec.", "success");
        window.location.href = "/home";
    }

    async function login(identifier, password) {
        const payload = identifier.includes("@")
            ? { email: identifier, password }
            : { username: identifier, password };

        const data = await apiFetch("/auth/login", {
            method: "POST",
            body: JSON.stringify(payload)
        });
        setSession(data);
        showToast("Signed in successfully.", "success");
        window.location.href = "/home";
    }

    async function checkAuth() {
        if (!token()) {
            redirectToLogin();
            return null;
        }

        try {
            const user = await apiFetch("/auth/me", {
                headers: { Authorization: `Bearer ${token()}` }
            });
            localStorage.setItem(storage.userId, user.id);
            localStorage.setItem(storage.username, user.username);
            updateAuthUi();
            return user;
        } catch (error) {
            logout(false);
            return null;
        }
    }

    function logout(showMessage = true) {
        clearSession();
        if (showMessage) showToast("You have been signed out.", "success");
        window.location.href = "/login";
    }

    async function performSearch(query, target = {}) {
        const value = String(query || "").trim();
        const containerId = target.containerId || "searchResults";
        const stateId = target.stateId || "searchState";
        const sectionId = target.sectionId || "searchSection";
        const titleId = target.titleId || "searchTitle";

        if (!value) return;

        const section = document.getElementById(sectionId);
        if (section) section.classList.remove("hidden");

        const title = document.getElementById(titleId);
        if (title) title.textContent = `Results for "${value}"`;

        renderSkeleton(containerId, 4);
        setState(stateId, "Searching the catalog...");

        try {
            const data = await apiFetch(`${API_BASE}/search?q=${encodeURIComponent(value)}`);
            renderMovieGrid(containerId, data.results, {
                emptyStateId: stateId,
                emptyMessage: `No movies found for "${value}".`
            });
        } catch (error) {
            renderMovieGrid(containerId, [], { emptyStateId: stateId, emptyMessage: "" });
            setState(stateId, error.message || "Search failed.", "error");
        }
    }

    async function loadHomePage() {
        const user = await checkAuth();
        if (!user) return;

        const id = userId();
        const pending = sessionStorage.getItem(storage.pendingSearch);
        if (pending) {
            sessionStorage.removeItem(storage.pendingSearch);
            const input = document.getElementById("navSearchInput");
            if (input) input.value = pending;
            performSearch(pending);
        }

        renderSkeleton("recommendedRow", 6);
        renderSkeleton("popularRow", 6);
        renderSkeleton("topRatedRow", 6);

        try {
            const recData = await authFetch(`${API_BASE}/recommend/${id}`);
            const recs = recData?.recommendations || [];
            renderMovieGrid("recommendedRow", recs, { emptyMessage: "No personalized recommendations yet." });
            setHero(recs[0]);
        } catch (error) {
            renderMovieGrid("recommendedRow", [], { emptyMessage: error.message || "Could not load recommendations." });
        }

        try {
            const popular = await apiFetch(`${API_BASE}/popular`);
            renderMovieGrid("popularRow", popular.recommendations, { emptyMessage: "Popular movies are unavailable." });
        } catch (error) {
            renderMovieGrid("popularRow", [], { emptyMessage: error.message || "Could not load popular movies." });
        }

        try {
            const topRated = await apiFetch(`${API_BASE}/top-rated`);
            renderMovieGrid("topRatedRow", topRated.recommendations, { emptyMessage: "Top rated movies are unavailable." });
        } catch (error) {
            renderMovieGrid("topRatedRow", [], { emptyMessage: error.message || "Could not load top rated movies." });
        }
    }

    function setHero(movie) {
        const hero = document.getElementById("hero");
        const title = document.getElementById("heroTitle");
        const desc = document.getElementById("heroDesc");
        const info = document.getElementById("heroInfoBtn");

        if (!hero || !title || !desc || !info) return;

        if (!movie) {
            title.textContent = "Discover movies built around your taste";
            desc.textContent = "Browse popular and top-rated movies while recommendations warm up.";
            info.disabled = true;
            return;
        }

        title.textContent = movieTitle(movie);
        desc.textContent = normalizeGenres(movie.genres).join(", ") || "Recommended for you";
        if (movie.poster_path) {
            hero.style.backgroundImage = `url("${posterUrl(movie.poster_path, "original")}")`;
        }
        info.disabled = false;
        info.onclick = () => {
            window.location.href = `/movie/${movie.movieId}`;
        };
    }

    async function loadMovieDetailPage() {
        const user = await checkAuth();
        if (!user) return;

        const page = $(".detail-page");
        const movieId = page?.dataset.movieId;
        if (!movieId) return;

        try {
            const movie = await apiFetch(`${API_BASE}/movie/${movieId}`);
            renderMovieDetail(movie);
            setState("detailState", "");
            $("#detailContent")?.classList.remove("hidden");
        } catch (error) {
            setState("detailState", error.message || "Could not load movie details.", "error");
        }

        try {
            const similar = await apiFetch(`${API_BASE}/similar/${movieId}`);
            renderMovieGrid("similarRow", similar.similar_movies, {
                emptyStateId: "similarState",
                emptyMessage: "No similar movies were found."
            });
        } catch (error) {
            renderMovieGrid("similarRow", [], { emptyStateId: "similarState", emptyMessage: "" });
            setState("similarState", error.message || "Could not load similar movies.", "error");
        }
    }

    function renderMovieDetail(movie) {
        const title = movieTitle(movie);
        const poster = $("#moviePoster");
        if (poster) {
            poster.src = posterUrl(movie.poster_path, "w500");
            poster.alt = `${title} poster`;
        }

        const titleEl = $("#movieTitle");
        if (titleEl) titleEl.textContent = title;

        const genresEl = $("#movieGenres");
        if (genresEl) {
            genresEl.innerHTML = "";
            normalizeGenres(movie.genres).forEach((genre) => {
                const chip = document.createElement("span");
                chip.className = "genre-chip";
                chip.textContent = genre;
                genresEl.appendChild(chip);
            });
        }

        const rating = $("#movieRating");
        if (rating) rating.textContent = `${movieRating(movie)} / 5 average rating`;
    }

    async function addToWatchlist(movieId) {
        try {
            await authFetch("/watchlist/add", {
                method: "POST",
                body: JSON.stringify({ movie_id: Number(movieId) })
            });
            showToast("Added to your watchlist.", "success");
        } catch (error) {
            showToast(error.message || "Could not add to watchlist.", "error");
        }
    }

    async function rateMovie(movieId, rating) {
        try {
            await authFetch("/ratings/add", {
                method: "POST",
                body: JSON.stringify({ movie_id: Number(movieId), rating: Number(rating) })
            });
            $all(".rating-control button").forEach((button) => {
                button.classList.toggle("active", Number(button.dataset.rating) <= Number(rating));
            });
            showToast("Rating saved.", "success");
        } catch (error) {
            showToast(error.message || "Could not save rating.", "error");
        }
    }

    function bindForms() {
        $("#signupForm")?.addEventListener("submit", async (event) => {
            event.preventDefault();
            const form = event.currentTarget;
            const button = form.querySelector("button[type='submit']");
            button.disabled = true;
            try {
                await signup(form.username.value.trim(), form.email.value.trim(), form.password.value);
            } catch (error) {
                showToast(error.message || "Signup failed.", "error");
            } finally {
                button.disabled = false;
            }
        });

        $("#loginForm")?.addEventListener("submit", async (event) => {
            event.preventDefault();
            const form = event.currentTarget;
            const button = form.querySelector("button[type='submit']");
            button.disabled = true;
            try {
                await login(form.identifier.value.trim(), form.password.value);
            } catch (error) {
                showToast(error.message || "Login failed.", "error");
            } finally {
                button.disabled = false;
            }
        });

        $("#landingSearchForm")?.addEventListener("submit", (event) => {
            event.preventDefault();
            performSearch($("#landingSearchInput")?.value, {
                containerId: "landingSearchResults",
                stateId: "landingSearchState",
                sectionId: null,
                titleId: null
            });
        });

        $("#navSearchForm")?.addEventListener("submit", (event) => {
            event.preventDefault();
            const query = $("#navSearchInput")?.value.trim();
            if (!query) return;

            if (document.body.dataset.page === "home") {
                performSearch(query);
                return;
            }

            if (token()) {
                sessionStorage.setItem(storage.pendingSearch, query);
                window.location.href = "/home";
                return;
            }

            const landingInput = $("#landingSearchInput");
            if (landingInput) {
                landingInput.value = query;
                performSearch(query, {
                    containerId: "landingSearchResults",
                    stateId: "landingSearchState",
                    sectionId: null,
                    titleId: null
                });
            } else {
                window.location.href = "/";
            }
        });

        $all("[data-action='logout']").forEach((button) => {
            button.addEventListener("click", () => logout());
        });

        $("#heroRefreshBtn")?.addEventListener("click", () => {
            loadHomePage();
            showToast("Refreshing your shelves.", "success");
        });

        $("#watchlistBtn")?.addEventListener("click", () => {
            const movieId = $(".detail-page")?.dataset.movieId;
            if (movieId) addToWatchlist(movieId);
        });

        $all(".rating-control button").forEach((button) => {
            button.addEventListener("click", () => {
                const movieId = $(".detail-page")?.dataset.movieId;
                if (movieId) rateMovie(movieId, button.dataset.rating);
            });
        });
    }

    function bootPage() {
        updateAuthUi();
        bindForms();

        if (document.body.dataset.requiresAuth === "true") {
            checkAuth();
        }

        if (document.body.dataset.page === "landing" && token()) {
            window.location.href = "/home";
        }

        if (document.body.dataset.page === "login" && token()) {
            window.location.href = "/home";
        }

        if (document.body.dataset.page === "signup" && token()) {
            window.location.href = "/home";
        }

        if (document.body.dataset.page === "home") {
            loadHomePage();
        }

        if (document.body.dataset.page === "movie-detail") {
            loadMovieDetailPage();
        }
    }

    window.MovieRec = {
        authFetch,
        apiFetch,
        checkAuth,
        logout,
        performSearch,
        renderMovieGrid,
        addToWatchlist,
        rateMovie
    };

    document.addEventListener("DOMContentLoaded", bootPage);
})();
