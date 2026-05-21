const API_BASE = '/api';

// Auth State
let currentUser = null;
const token = localStorage.getItem('access_token');

// Helper for authenticated requests
async function authFetch(url, options = {}) {
    const token = localStorage.getItem('access_token');
    const headers = {
        'Content-Type': 'application/json',
        ...options.headers
    };
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }
    
    const response = await fetch(url, { ...options, headers });
    if (response.status === 401) {
        logout();
        return null;
    }
    return response;
}

// Auth Functions
async function signup(username, email, password) {
    const res = await fetch('/auth/signup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, email, password })
    });
    const data = await res.json();
    if (res.ok) {
        localStorage.setItem('access_token', data.access_token);
        localStorage.setItem('user_id', data.user_id);
        localStorage.setItem('username', data.username);
        window.location.href = '/home';
    } else {
        alert(data.detail || 'Signup failed');
    }
}

async function login(email, password) {
    const res = await fetch('/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
    });
    const data = await res.json();
    if (res.ok) {
        localStorage.setItem('access_token', data.access_token);
        localStorage.setItem('user_id', data.user_id);
        localStorage.setItem('username', data.username);
        window.location.href = '/home';
    } else {
        alert(data.detail || 'Login failed');
    }
}

function logout() {
    localStorage.clear();
    window.location.href = '/login';
}

async function checkAuth() {
    const res = await authFetch('/auth/me');
    if (!res || !res.ok) {
        if (!['/login', '/signup', '/'].includes(window.location.pathname)) {
            window.location.href = '/login';
        }
        return null;
    }
    const user = await res.json();
    document.querySelectorAll('.username-display').forEach(el => el.textContent = user.username);
    return user;
}

// Movie Data Functions
async function fetchMovies(endpoint) {
    const res = await authFetch(endpoint);
    if (res && res.ok) return await res.json();
    return { recommendations: [] };
}

function renderMovieRow(containerId, movies) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    container.innerHTML = movies.map(m => `
        <div class="movie-card" onclick="window.location.href='/movie/${m.movieId}'">
            <img src="${m.poster_path ? 'https://image.tmdb.org/t/p/w342' + m.poster_path : '/static/placeholder.jpg'}" 
                 onerror="this.src='https://via.placeholder.com/200x300?text=${encodeURIComponent(m.title)}'"
                 alt="${m.title}">
            <div class="card-overlay">
                <div>${m.title}</div>
                <div style="font-size: 0.8rem; color: #aaa;">⭐ ${m.global_avg_rating}</div>
            </div>
        </div>
    `).join('');
}

// Action Functions
async function addToWatchlist(movieId) {
    const res = await authFetch('/watchlist/add', {
        method: 'POST',
        body: JSON.stringify({ movie_id: movieId })
    });
    if (res.ok) alert('Added to watchlist!');
}

async function rateMovie(movieId, rating) {
    const res = await authFetch('/ratings/add', {
        method: 'POST',
        body: JSON.stringify({ movie_id: movieId, rating })
    });
    if (res.ok) alert('Rating saved!');
}

// Global Search
async function performSearch(query) {
    if (!query) return;
    const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
    const data = await res.json();
    renderMovieRow('searchResults', data.results);
}
