-- List all shows with at least one genre linked. Each record should display: tv_shows.title - tv_show_genres.genre_id, sorted by tv_shows.title and tv_show_genres.genre_id in ascending order.
SELECT tv_shows.title, tv_show_genres.genre_id FROM tv_shows, tv_show_genres
WHERE tv_shows.id = tv_show_genres.show_id
ORDER BY tv_shows.title ASC, tv_show_genres.genre_id ASC;
